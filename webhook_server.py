"""
Stripe Webhook Server (Production Ready)

Handles incoming Stripe webhooks to process payments and add credits.

Deployment:
    Railway: Connect GitHub repo, set env vars, deploy
    Fly.io: fly launch && fly deploy
    Docker: docker build -f Dockerfile.webhook -t webhook . && docker run -p 8080:8080 webhook
"""

import os
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify
import stripe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================
# Configuration (from environment variables)
# ============================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# Credit package mapping: price_id -> credits
CREDIT_PACKAGES = {
    os.getenv("STRIPE_PRICE_10_CREDITS", ""): 10,
    os.getenv("STRIPE_PRICE_50_CREDITS", ""): 50,
    os.getenv("STRIPE_PRICE_100_CREDITS", ""): 100,
}

# Initialize Stripe
stripe.api_key = STRIPE_SECRET_KEY

# ============================================
# Database (Supabase) - Inline for single-file deployment
# ============================================

_supabase_client = None

def get_db():
    """Get or create Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


def get_or_create_user(discord_id: str, username: str = None) -> dict:
    """Get existing user or create new one."""
    db = get_db()
    result = db.table("users").select("*").eq("discord_id", discord_id).execute()
    
    if result.data:
        return result.data[0]
    
    new_user = {
        "discord_id": discord_id,
        "username": username,
        "credits": 0,
        "total_generations": 0,
        "created_at": datetime.utcnow().isoformat()
    }
    result = db.table("users").insert(new_user).execute()
    return result.data[0]


def add_credits(discord_id: str, amount: int, username: str = None) -> int:
    """Add credits to user. Returns new balance."""
    db = get_db()
    user = get_or_create_user(discord_id, username)
    new_balance = user["credits"] + amount
    
    db.table("users").update({
        "credits": new_balance
    }).eq("discord_id", discord_id).execute()
    
    logger.info(f"Added {amount} credits to user {discord_id}. New balance: {new_balance}")
    return new_balance


def get_payment_by_stripe_id(stripe_payment_id: str) -> dict | None:
    """Check if payment already processed (idempotency)."""
    db = get_db()
    result = db.table("payments").select("*").eq("stripe_payment_id", stripe_payment_id).execute()
    return result.data[0] if result.data else None


def log_payment(discord_id: str, stripe_payment_id: str, amount_cents: int, credits_added: int) -> dict:
    """Log a successful payment."""
    db = get_db()
    record = {
        "discord_id": discord_id,
        "stripe_payment_id": stripe_payment_id,
        "amount_cents": amount_cents,
        "credits_added": credits_added,
        "status": "completed",
        "created_at": datetime.utcnow().isoformat()
    }
    result = db.table("payments").insert(record).execute()
    return result.data[0]


# ============================================
# Simple Rate Limiting (in-memory)
# ============================================

rate_limit_store = {}  # IP -> list of timestamps

def rate_limit(max_requests: int = 30, window_seconds: int = 60):
    """Rate limit decorator."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            ip = request.headers.get('X-Forwarded-For', request.remote_addr)
            if ip:
                ip = ip.split(',')[0].strip()
            
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window_seconds)
            
            # Clean old entries
            if ip in rate_limit_store:
                rate_limit_store[ip] = [t for t in rate_limit_store[ip] if t > window_start]
            else:
                rate_limit_store[ip] = []
            
            # Check limit
            if len(rate_limit_store[ip]) >= max_requests:
                logger.warning(f"Rate limit exceeded for IP: {ip}")
                return jsonify({"error": "Rate limit exceeded"}), 429
            
            # Record request
            rate_limit_store[ip].append(now)
            
            return f(*args, **kwargs)
        return wrapped
    return decorator


# ============================================
# Webhook Endpoint
# ============================================

@app.route("/webhook/stripe", methods=["POST"])
@rate_limit(max_requests=60, window_seconds=60)  # 60 requests per minute
def stripe_webhook():
    """Handle Stripe webhook events."""
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature")
    
    logger.info("Received webhook request")
    
    # Verify webhook signature
    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except ValueError as e:
            logger.error(f"Invalid payload: {e}")
            return jsonify({"error": "Invalid payload"}), 400
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid signature: {e}")
            return jsonify({"error": "Invalid signature"}), 400
    else:
        # No webhook secret - INSECURE, dev only
        import json
        event = json.loads(payload)
        logger.warning("⚠️ Webhook signature verification DISABLED - set STRIPE_WEBHOOK_SECRET!")
    
    event_type = event.get("type")
    logger.info(f"Event type: {event_type}")
    
    # Handle checkout completion
    if event_type == "checkout.session.completed":
        session = event["data"]["object"]
        process_successful_payment(session)
    
    return jsonify({"status": "success"}), 200


def process_successful_payment(session: dict):
    """Process a successful payment and add credits."""
    payment_id = session.get("id") or session.get("payment_intent")
    
    # Idempotency check
    if get_payment_by_stripe_id(payment_id):
        logger.info(f"Payment {payment_id} already processed, skipping")
        return
    
    # Get Discord user ID
    discord_id = session.get("client_reference_id")
    if not discord_id:
        discord_id = session.get("metadata", {}).get("discord_id")
    
    if not discord_id:
        logger.error(f"No discord_id found in payment {payment_id}")
        logger.error(f"Session keys: {list(session.keys())}")
        return
    
    # Determine credits
    credits_to_add = 0
    amount_cents = session.get("amount_total", 0)
    
    # From metadata
    metadata = session.get("metadata", {})
    if "credits" in metadata:
        credits_to_add = int(metadata["credits"])
    
    # From line items / price ID
    if credits_to_add == 0:
        try:
            full_session = stripe.checkout.Session.retrieve(
                session["id"],
                expand=["line_items.data.price"]
            )
            for item in full_session.line_items.data:
                price_id = item.price.id
                # Check our mapping
                if price_id in CREDIT_PACKAGES:
                    credits_to_add += CREDIT_PACKAGES[price_id] * item.quantity
                # Check price metadata
                elif item.price.metadata.get("credits"):
                    credits_to_add += int(item.price.metadata["credits"]) * item.quantity
        except Exception as e:
            logger.error(f"Error fetching session details: {e}")
    
    if credits_to_add == 0:
        logger.error(f"Could not determine credits for payment {payment_id}")
        return
    
    # Add credits
    new_balance = add_credits(discord_id, credits_to_add)
    
    # Log payment
    log_payment(
        discord_id=discord_id,
        stripe_payment_id=payment_id,
        amount_cents=amount_cents,
        credits_added=credits_to_add
    )
    
    logger.info(f"✅ Payment processed: {credits_to_add} credits -> user {discord_id} (balance: {new_balance})")


# ============================================
# Health & Info Endpoints
# ============================================

@app.route("/health", methods=["GET"])
def health():
    """Health check for uptime monitoring."""
    # Verify DB connection
    try:
        get_db()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "healthy",
        "database": db_status,
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "webhook_secret_configured": bool(STRIPE_WEBHOOK_SECRET)
    }), 200


@app.route("/", methods=["GET"])
def index():
    """Root endpoint."""
    return jsonify({
        "service": "Discord Image Bot - Stripe Webhook Server",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "webhook": "POST /webhook/stripe",
            "health": "GET /health"
        }
    })


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    
    logger.info(f"Starting webhook server on port {port}")
    logger.info(f"Stripe configured: {bool(STRIPE_SECRET_KEY)}")
    logger.info(f"Webhook secret configured: {bool(STRIPE_WEBHOOK_SECRET)}")
    logger.info(f"Supabase configured: {bool(SUPABASE_URL and SUPABASE_KEY)}")
    
    # For local development only - use gunicorn in production
    app.run(host="0.0.0.0", port=port, debug=False)
