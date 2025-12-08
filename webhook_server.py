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
from pathlib import Path
from flask import Flask, request, jsonify
import stripe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================
# Credit Transaction Logging
# ============================================

class CreditLogger:
    """Handles logging of all credit transactions to a file."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "credits_logs.txt"

    def log_transaction(
        self,
        discord_id: str,
        action: str,
        amount: int,
        model: str = None,
        status: str = None,
        details: str = None
    ):
        """Log a credit transaction."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Build log entry
        parts = [
            timestamp,
            discord_id,
            action,
            str(amount)
        ]

        if model:
            parts.append(f"model={model}")
        if status:
            parts.append(f"status={status}")
        if details:
            parts.append(f"details={details}")

        log_entry = " | ".join(parts)

        # Append to log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

# Global credit logger instance
credit_logger = CreditLogger()

app = Flask(__name__)

# ============================================
# Configuration (from environment variables)
# ============================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Subscription tier mapping: price_id -> tier info
# Basic: $14/month = 58,800 credits
# Pro: $26/month = 133,224 credits (+22% bonus)
# Creator: $44/month = 308,616 credits (+67% bonus)
# Unlimited: $98/month = unlimited credits
SUBSCRIPTION_TIERS = {
    os.getenv("STRIPE_PRICE_BASIC", ""): {
        "tier": "basic",
        "credits": 58800,
        "name": "Basic Plan"
    },
    os.getenv("STRIPE_PRICE_PRO", ""): {
        "tier": "pro",
        "credits": 133224,
        "name": "Pro Plan"
    },
    os.getenv("STRIPE_PRICE_CREATOR", ""): {
        "tier": "creator",
        "credits": 308616,
        "name": "Creator Plan"
    },
    os.getenv("STRIPE_PRICE_UNLIMITED", ""): {
        "tier": "unlimited",
        "credits": -1,  # -1 represents unlimited
        "name": "Unlimited Plan"
    }
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


def add_credits(discord_id: str, amount: int, username: str = None, source: str = "PURCHASE") -> int:
    """Add credits to user. Returns new balance."""
    db = get_db()
    user = get_or_create_user(discord_id, username)
    new_balance = user["credits"] + amount

    db.table("users").update({
        "credits": new_balance
    }).eq("discord_id", discord_id).execute()

    # Log the transaction
    credit_logger.log_transaction(
        discord_id=discord_id,
        action=source,
        amount=amount,
        status="SUCCESS"
    )

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
# Subscription Management
# ============================================

def create_or_update_subscription(
    discord_id: str,
    subscription_id: str,
    status: str,
    tier: str,
    current_period_start: int,
    current_period_end: int,
    stripe_customer_id: str,
    cancel_at_period_end: bool = False
) -> dict:
    """
    Create or update subscription for a user.
    Also sets credits to tier allowance.
    """
    db = get_db()

    # Get credits for tier
    tier_credits = {
        "basic": 58800,
        "pro": 133224,
        "creator": 308616,
        "unlimited": -1  # -1 represents unlimited
    }
    credits = tier_credits.get(tier, 0)

    # Convert Unix timestamps to ISO format
    period_start = datetime.fromtimestamp(current_period_start).isoformat()
    period_end = datetime.fromtimestamp(current_period_end).isoformat()

    result = db.table("users").update({
        "subscription_id": subscription_id,
        "subscription_status": status,
        "subscription_tier": tier,
        "current_period_start": period_start,
        "current_period_end": period_end,
        "stripe_customer_id": stripe_customer_id,
        "cancel_at_period_end": cancel_at_period_end,
        "credits": credits,
        "updated_at": datetime.utcnow().isoformat()
    }).eq("discord_id", discord_id).execute()

    if result.data:
        credit_logger.log_transaction(
            discord_id=discord_id,
            action="SUBSCRIPTION_CREATED",
            amount=credits,
            status="SUCCESS",
            details=f"{tier} tier subscription"
        )
        logger.info(f"Subscription created/updated for {discord_id}: {tier} tier, {credits} credits")
        return result.data[0]
    else:
        logger.error(f"Failed to create/update subscription for {discord_id}")
        return None


def reset_credits_to_tier_allowance(discord_id: str) -> dict:
    """Reset user's credits to their subscription tier allowance."""
    db = get_db()
    user = get_or_create_user(discord_id)
    tier = user.get("subscription_tier")

    if not tier:
        logger.error(f"Cannot reset credits for {discord_id}: No subscription tier")
        return None

    # Get credits for tier
    tier_credits = {
        "basic": 58800,
        "pro": 133224,
        "creator": 308616,
        "unlimited": -1  # -1 represents unlimited
    }
    credits = tier_credits.get(tier, 0)

    # Reset credits
    result = db.table("users").update({
        "credits": credits,
        "updated_at": datetime.utcnow().isoformat()
    }).eq("discord_id", discord_id).execute()

    if result.data:
        credit_logger.log_transaction(
            discord_id=discord_id,
            action="CREDIT_RESET",
            amount=credits,
            status="SUCCESS",
            details=f"Credits reset to {tier} tier allowance"
        )
        logger.info(f"Credits reset for {discord_id}: {tier} tier → {credits} credits")
        return result.data[0]
    else:
        logger.error(f"Failed to reset credits for {discord_id}")
        return None


def suspend_subscription(discord_id: str) -> dict:
    """Suspend user's subscription (failed payment)."""
    db = get_db()
    result = db.table("users").update({
        "subscription_status": "suspended",
        "credits": 0,
        "updated_at": datetime.utcnow().isoformat()
    }).eq("discord_id", discord_id).execute()

    if result.data:
        credit_logger.log_transaction(
            discord_id=discord_id,
            action="SUBSCRIPTION_SUSPENDED",
            amount=0,
            status="SUCCESS",
            details="Subscription suspended"
        )
        logger.info(f"Subscription suspended for {discord_id}")
        return result.data[0]
    else:
        logger.error(f"Failed to suspend subscription for {discord_id}")
        return None


def cancel_subscription_db(discord_id: str, immediate: bool = False) -> dict:
    """Cancel user's subscription in database."""
    db = get_db()
    updates = {
        "updated_at": datetime.utcnow().isoformat()
    }

    if immediate:
        updates["subscription_status"] = "canceled"
        updates["credits"] = 0
        updates["cancel_at_period_end"] = False
    else:
        updates["cancel_at_period_end"] = True

    result = db.table("users").update(updates).eq("discord_id", discord_id).execute()

    if result.data:
        credit_logger.log_transaction(
            discord_id=discord_id,
            action="SUBSCRIPTION_CANCELED",
            amount=0,
            status="SUCCESS",
            details=f"Subscription canceled (immediate={immediate})"
        )
        logger.info(f"Subscription canceled for {discord_id} (immediate={immediate})")
        return result.data[0]
    else:
        logger.error(f"Failed to cancel subscription for {discord_id}")
        return None


def send_discord_notification(
    discord_id: str,
    credits_added: int,
    amount_cents: int,
    new_balance: int,
    username: str = None
):
    """Send payment notification to Discord webhook."""
    if not DISCORD_WEBHOOK_URL:
        logger.warning("Discord webhook URL not configured, skipping notification")
        return

    try:
        import requests

        # Format amount as dollars
        amount_dollars = amount_cents / 100

        # Build embed
        embed = {
            "title": "💳 Payment Received",
            "color": 0x00FF00,  # Green
            "fields": [
                {
                    "name": "User",
                    "value": f"<@{discord_id}>",
                    "inline": True
                },
                {
                    "name": "Amount Paid",
                    "value": f"${amount_dollars:.2f}",
                    "inline": True
                },
                {
                    "name": "Credits Added",
                    "value": f"{credits_added:,}",
                    "inline": True
                },
                {
                    "name": "New Balance",
                    "value": f"{new_balance:,} credits",
                    "inline": True
                },
                {
                    "name": "Discord ID",
                    "value": discord_id,
                    "inline": True
                }
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "Stripe Payment Webhook"
            }
        }

        # Add username if available
        if username:
            embed["fields"].insert(1, {
                "name": "Username",
                "value": username,
                "inline": True
            })

        # Send to Discord
        payload = {
            "embeds": [embed]
        }

        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json=payload,
            timeout=10
        )

        if response.status_code == 204:
            logger.info("✅ Discord notification sent successfully")
        else:
            logger.warning(f"Discord webhook returned status {response.status_code}")

    except Exception as e:
        # Don't fail payment processing if Discord notification fails
        logger.error(f"Failed to send Discord notification: {e}")


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

    # Handle subscription checkout completion
    if event_type == "checkout.session.completed":
        session = event["data"]["object"]

        # Check if it's a subscription checkout
        if session.get("mode") == "subscription":
            handle_subscription_checkout(session)
        else:
            # Old one-time payment (should not happen anymore)
            logger.warning(f"[Deprecated] Received one-time payment checkout: {session['id']}")
            process_successful_payment(session)  # Keep for backwards compatibility

    # Handle subscription creation
    elif event_type == "customer.subscription.created":
        subscription = event["data"]["object"]
        handle_subscription_created(subscription)

    # Handle subscription updates (tier changes, etc.)
    elif event_type == "customer.subscription.updated":
        subscription = event["data"]["object"]
        handle_subscription_updated(subscription)

    # Handle subscription deletion (cancellation)
    elif event_type == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        handle_subscription_deleted(subscription)

    # Handle successful invoice payment (monthly renewal)
    elif event_type == "invoice.payment_succeeded":
        invoice = event["data"]["object"]

        # Only process subscription invoices
        if invoice.get("billing_reason") == "subscription_cycle":
            handle_subscription_renewal(invoice)

    # Handle failed invoice payment
    elif event_type == "invoice.payment_failed":
        invoice = event["data"]["object"]
        handle_payment_failed(invoice)

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

    # Send Discord notification
    send_discord_notification(
        discord_id=discord_id,
        credits_added=credits_to_add,
        amount_cents=amount_cents,
        new_balance=new_balance
    )


# ============================================
# Subscription Event Handlers
# ============================================

def handle_subscription_checkout(session: dict):
    """Handle successful subscription checkout."""
    discord_id = session.get("client_reference_id")
    subscription_id = session.get("subscription")
    customer_id = session.get("customer")

    if not discord_id:
        logger.error(f"[Subscription] No client_reference_id in session: {session['id']}")
        return

    # Retrieve full subscription details from Stripe
    subscription = stripe.Subscription.retrieve(subscription_id)

    # Get price ID to determine tier
    price_id = subscription["items"]["data"][0]["price"]["id"]
    tier_info = SUBSCRIPTION_TIERS.get(price_id)

    if not tier_info:
        logger.error(f"[Subscription] Unknown price ID: {price_id}")
        return

    # Create subscription in database
    create_or_update_subscription(
        discord_id=discord_id,
        subscription_id=subscription_id,
        status=subscription["status"],
        tier=tier_info["tier"],
        current_period_start=subscription["current_period_start"],
        current_period_end=subscription["current_period_end"],
        stripe_customer_id=customer_id,
        cancel_at_period_end=subscription["cancel_at_period_end"]
    )

    logger.info(f"[Subscription] Created {tier_info['name']} for Discord ID: {discord_id}")


def handle_subscription_created(subscription: dict):
    """Handle subscription.created event."""
    customer_id = subscription["customer"]

    # Get discord_id from customer metadata
    customer = stripe.Customer.retrieve(customer_id)
    discord_id = customer.get("metadata", {}).get("discord_id")

    if not discord_id:
        logger.error(f"[Subscription] No discord_id in customer metadata: {customer_id}")
        return

    # Get tier from price ID
    price_id = subscription["items"]["data"][0]["price"]["id"]
    tier_info = SUBSCRIPTION_TIERS.get(price_id)

    if not tier_info:
        logger.error(f"[Subscription] Unknown price ID: {price_id}")
        return

    # Create/update subscription
    create_or_update_subscription(
        discord_id=discord_id,
        subscription_id=subscription["id"],
        status=subscription["status"],
        tier=tier_info["tier"],
        current_period_start=subscription["current_period_start"],
        current_period_end=subscription["current_period_end"],
        stripe_customer_id=customer_id,
        cancel_at_period_end=subscription["cancel_at_period_end"]
    )

    logger.info(f"[Subscription] Created for Discord ID: {discord_id}")


def handle_subscription_updated(subscription: dict):
    """Handle subscription.updated event (tier changes, status changes)."""
    customer_id = subscription["customer"]

    # Get discord_id from customer metadata
    customer = stripe.Customer.retrieve(customer_id)
    discord_id = customer.get("metadata", {}).get("discord_id")

    if not discord_id:
        logger.error(f"[Subscription] No discord_id in customer metadata: {customer_id}")
        return

    # Get tier from price ID
    price_id = subscription["items"]["data"][0]["price"]["id"]
    tier_info = SUBSCRIPTION_TIERS.get(price_id)

    # Update subscription
    create_or_update_subscription(
        discord_id=discord_id,
        subscription_id=subscription["id"],
        status=subscription["status"],
        tier=tier_info["tier"] if tier_info else "unknown",
        current_period_start=subscription["current_period_start"],
        current_period_end=subscription["current_period_end"],
        stripe_customer_id=customer_id,
        cancel_at_period_end=subscription["cancel_at_period_end"]
    )

    logger.info(f"[Subscription] Updated for Discord ID: {discord_id}")


def handle_subscription_deleted(subscription: dict):
    """Handle subscription.deleted event (cancellation)."""
    customer_id = subscription["customer"]

    # Get discord_id from customer metadata
    customer = stripe.Customer.retrieve(customer_id)
    discord_id = customer.get("metadata", {}).get("discord_id")

    if not discord_id:
        logger.error(f"[Subscription] No discord_id in customer metadata: {customer_id}")
        return

    # Suspend subscription (set credits to 0, status to canceled)
    cancel_subscription_db(discord_id, immediate=True)

    logger.info(f"[Subscription] Deleted for Discord ID: {discord_id}")


def handle_subscription_renewal(invoice: dict):
    """Handle successful subscription renewal (monthly reset)."""
    customer_id = invoice["customer"]
    subscription_id = invoice["subscription"]

    # Get discord_id from customer metadata
    customer = stripe.Customer.retrieve(customer_id)
    discord_id = customer.get("metadata", {}).get("discord_id")

    if not discord_id:
        logger.error(f"[Subscription] No discord_id in customer metadata: {customer_id}")
        return

    # Reset credits to tier allowance
    reset_credits_to_tier_allowance(discord_id)

    logger.info(f"[Subscription] Renewal processed for Discord ID: {discord_id}")


def handle_payment_failed(invoice: dict):
    """Handle failed payment (suspend immediately)."""
    customer_id = invoice["customer"]

    # Get discord_id from customer metadata
    customer = stripe.Customer.retrieve(customer_id)
    discord_id = customer.get("metadata", {}).get("discord_id")

    if not discord_id:
        logger.error(f"[Subscription] No discord_id in customer metadata: {customer_id}")
        return

    # Suspend subscription immediately
    suspend_subscription(discord_id)

    logger.info(f"[Subscription] Suspended due to failed payment for Discord ID: {discord_id}")


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
        "webhook_secret_configured": bool(STRIPE_WEBHOOK_SECRET),
        "discord_webhook_configured": bool(DISCORD_WEBHOOK_URL)
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
    logger.info(f"Discord webhook configured: {bool(DISCORD_WEBHOOK_URL)}")
    
    # For local development only - use gunicorn in production
    app.run(host="0.0.0.0", port=port, debug=False)
