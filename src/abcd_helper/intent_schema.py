"""
Intent Schema Specification for ABCD Conversational Analytics.

Defines the normative ground-truth slot requirements for all 55 customer intents
(across 10 conversation flows) derived from the dataset SOP guidelines, KB action
dependencies, and domain ontology.

Mandatory customer slots are represented in Conjunctive Normal Form (CNF):
- Outer list: Logical AND (all slot groups must be satisfied).
- Inner list: Logical OR (alternative slot options, e.g. identifiers or addresses).
"""

import pandas as pd

from abcd_helper.config import INTENT_SCHEMA_PARQUET

# =============================================================================
# PROVENANCE NOTICE: AI-GENERATED NORMATIVE INTENT SCHEMA SPECIFICATION
# =============================================================================
# The INTENT_SCHEMA_DATA dictionary below was initially generated via LLM normative
# inference (Gemini 3.8 Flash - medium) and subsequently audited and refined by
# Gemini 4 Argon - medium by synthesizing three authoritative raw dataset files:
#   1. data/raw/guidelines.json: Human-written SOP flowcharts, customer authentication
#      policies, prerequisite steps, and branching business rules per subflow.
#   2. data/raw/ontology.json: Formal schema defining the 30 discrete system actions
#      and their required/accepted slot arguments.
#   3. data/raw/kb.json: Canonical action execution sequences for all 55 intents.
#
# Audit & Refinement Notes (Gemini 4 Argon - medium):
#   - Excluded the untagged `pull-up-account` opening clause (`["customer_name", "account_id"]`)
#     from `mandatory_slots` because `customer_name` is a synthetic dummy entity not
#     annotated in ABCD's ground-truth `slot_tags` (and used in 97.1% of `pull-up-account`
#     calls), ensuring mandatory CNF evaluation aligns strictly with benchmarked DST slots.
#   - Expanded truncated OR-clauses in `refund_initiate` (`["full_address", "street_address", "account_id"]`
#     for `enter-details`) and the 4 `manage_account` verification subflows (`["zip_code", "phone",
#     "pin_number", "username", "email", "order_id"]` for 3-of-7 `verify-identity`) to match
#     `guidelines.json` branching rules.
#
# Rationale: Raw ABCD defines slot requirements only at the atomic action level
# in ontology.json, lacking an intent-level ground truth mapping customer intents
# to required customer information. This specification establishes that ground truth
# by structuring customer requirements in Conjunctive Normal Form (CNF) and
# separating them from agent operational parameters.
# =============================================================================

# Comprehensive normative definitions for all 55 subflows
INTENT_SCHEMA_DATA = [
    # =========================================================================
    # 1. ACCOUNT ACCESS (3 subflows)
    # =========================================================================
    {
        "flow": "account_access",
        "intent": "recover_username",
        "canonical_actions": ["pull-up-account", "verify-identity"],
        "mandatory_slots": [
            ["zip_code"],
            ["email", "phone"],
        ],
        "agent_action_slots": [],
        "sop_summary": (
            "Customer lost username. Pull up account using customer name or account ID, "
            "verify customer identity with postal code and email/phone, then provide username."
        ),
    },
    {
        "flow": "account_access",
        "intent": "recover_password",
        "canonical_actions": ["pull-up-account", "enter-details", "make-password"],
        "mandatory_slots": [
            ["username"],
        ],
        "agent_action_slots": ["details_slotval"],
        "sop_summary": (
            "Customer forgot password. Pull up account by name or account ID, confirm "
            "username via enter-details, and generate a temporary password."
        ),
    },
    {
        "flow": "account_access",
        "intent": "reset_2fa",
        "canonical_actions": ["pull-up-account", "enter-details", "send-link"],
        "mandatory_slots": [
            ["email"],
        ],
        "agent_action_slots": ["details_slotval"],
        "sop_summary": (
            "Customer needs two-factor authentication reset. Pull up account by name or "
            "account ID, get customer email to send security link."
        ),
    },
    # =========================================================================
    # 2. MANAGE ACCOUNT (8 subflows)
    # =========================================================================
    {
        "flow": "manage_account",
        "intent": "status_service_added",
        "canonical_actions": [
            "pull-up-account",
            "record-reason",
            "ask-the-oracle",
            "update-account",
            "offer-refund",
        ],
        "mandatory_slots": [],
        "agent_action_slots": ["reason_slotval", "account_slotval", "amount"],
        "sop_summary": (
            "Customer inquires about unauthorized service added. Pull up account, record "
            "reason, query oracle, remove service via update-account, and offer refund if appropriate."
        ),
    },
    {
        "flow": "manage_account",
        "intent": "status_service_removed",
        "canonical_actions": [
            "pull-up-account",
            "record-reason",
            "ask-the-oracle",
            "update-account",
        ],
        "mandatory_slots": [],
        "agent_action_slots": ["reason_slotval", "account_slotval"],
        "sop_summary": (
            "Customer inquires about service removed. Pull up account, record reason, "
            "check oracle, and reinstate service via update-account."
        ),
    },
    {
        "flow": "manage_account",
        "intent": "status_shipping_question",
        "canonical_actions": ["pull-up-account", "ask-the-oracle", "send-link"],
        "mandatory_slots": [],
        "agent_action_slots": [],
        "sop_summary": (
            "Customer has general shipping status question on profile. Pull up account, "
            "consult oracle, and send informative link."
        ),
    },
    {
        "flow": "manage_account",
        "intent": "status_credit_missing",
        "canonical_actions": [
            "pull-up-account",
            "record-reason",
            "ask-the-oracle",
            "promo-code",
        ],
        "mandatory_slots": [],
        "agent_action_slots": ["reason_slotval"],
        "sop_summary": (
            "Customer states account credit is missing. Pull up account, record reason, "
            "verify oracle, issue replacement promo code if confirmed."
        ),
    },
    {
        "flow": "manage_account",
        "intent": "manage_change_address",
        "canonical_actions": [
            "pull-up-account",
            "record-reason",
            "verify-identity",
            "update-account",
        ],
        "mandatory_slots": [
            ["street_address", "full_address"],
            ["zip_code", "phone", "pin_number", "username", "email", "order_id"],
        ],
        "agent_action_slots": ["account_slotval"],
        "sop_summary": (
            "Customer updating profile address. Pull up account, record current address, "
            "verify customer identity, and update account with new address."
        ),
    },
    {
        "flow": "manage_account",
        "intent": "manage_change_name",
        "canonical_actions": [
            "pull-up-account",
            "record-reason",
            "verify-identity",
            "update-account",
        ],
        "mandatory_slots": [
            ["zip_code", "phone", "pin_number", "username", "email", "order_id"],
        ],
        "agent_action_slots": ["account_slotval"],
        "sop_summary": (
            "Customer changing legal name on account. Pull up account, record current name, "
            "verify identity, and update account."
        ),
    },
    {
        "flow": "manage_account",
        "intent": "manage_change_phone",
        "canonical_actions": [
            "pull-up-account",
            "record-reason",
            "verify-identity",
            "update-account",
        ],
        "mandatory_slots": [
            ["phone"],
            ["zip_code", "pin_number", "username", "email", "order_id"],
        ],
        "agent_action_slots": ["account_slotval"],
        "sop_summary": (
            "Customer updating contact phone number. Pull up account, record current phone, "
            "verify identity, and update account with new phone."
        ),
    },
    {
        "flow": "manage_account",
        "intent": "manage_payment_method",
        "canonical_actions": [
            "pull-up-account",
            "record-reason",
            "verify-identity",
            "update-account",
        ],
        "mandatory_slots": [
            ["payment_method"],
            ["zip_code", "phone", "pin_number", "username", "email", "order_id"],
        ],
        "agent_action_slots": ["account_slotval"],
        "sop_summary": (
            "Customer changing saved payment method. Pull up account, record current payment method, "
            "verify identity, and update account."
        ),
    },
    # =========================================================================
    # 3. ORDER ISSUE (8 subflows)
    # =========================================================================
    {
        "flow": "order_issue",
        "intent": "status_mystery_fee",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "ask-the-oracle",
            "membership",
            "update-order",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": ["membership_level", "order_slotval", "change_option"],
        "sop_summary": (
            "Customer questions unexplained fee on order. Pull up account, verify order, "
            "check oracle and membership privileges, and waive/update order fee."
        ),
    },
    {
        "flow": "order_issue",
        "intent": "status_delivery_time",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "ask-the-oracle",
            "shipping-status",
            "update-order",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": ["shipping_option", "order_slotval"],
        "sop_summary": (
            "Customer checking delivery estimate. Pull up account, verify order, "
            "check shipping status and oracle, adjust order delivery if delayed."
        ),
    },
    {
        "flow": "order_issue",
        "intent": "status_payment_method",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "shipping-status",
            "update-order",
        ],
        "mandatory_slots": [
            ["order_id"],
            ["payment_method"],
        ],
        "agent_action_slots": ["shipping_option", "order_slotval"],
        "sop_summary": (
            "Customer modifying payment method on existing order. Pull up account, "
            "verify order, check shipping status, update payment method on order."
        ),
    },
    {
        "flow": "order_issue",
        "intent": "status_quantity",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "ask-the-oracle",
            "shipping-status",
            "offer-refund",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": ["shipping_option", "amount"],
        "sop_summary": (
            "Customer received incorrect quantity. Pull up account, verify order, "
            "check oracle and shipping status, offer refund for missing item."
        ),
    },
    {
        "flow": "order_issue",
        "intent": "manage_upgrade",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "shipping-status",
            "membership",
            "update-order",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": [
            "shipping_option",
            "membership_level",
            "order_slotval",
            "change_option",
        ],
        "sop_summary": (
            "Customer requests expedited/overnight shipping upgrade. Pull up account, "
            "verify order, check shipping status and membership level, update order shipping."
        ),
    },
    {
        "flow": "order_issue",
        "intent": "manage_downgrade",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "shipping-status",
            "membership",
            "update-order",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": [
            "shipping_option",
            "membership_level",
            "order_slotval",
            "change_option",
        ],
        "sop_summary": (
            "Customer downgrades shipping speed. Pull up account, verify order, "
            "inspect shipping status, update order."
        ),
    },
    {
        "flow": "order_issue",
        "intent": "manage_create",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "shipping-status",
            "membership",
            "make-purchase",
        ],
        "mandatory_slots": [
            ["order_id"],
            ["product", "name"],
        ],
        "agent_action_slots": ["shipping_option", "membership_level"],
        "sop_summary": (
            "Customer adding item to existing order. Pull up account, verify order, "
            "verify shipping status and membership, create purchase item."
        ),
    },
    {
        "flow": "order_issue",
        "intent": "manage_cancel",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "shipping-status",
            "membership",
            "offer-refund",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": ["shipping_option", "membership_level", "amount"],
        "sop_summary": (
            "Customer canceling item/order. Pull up account, verify order, "
            "check shipping status and membership level, process cancellation refund."
        ),
    },
    # =========================================================================
    # 4. PRODUCT DEFECT (6 subflows)
    # =========================================================================
    {
        "flow": "product_defect",
        "intent": "refund_initiate",
        "canonical_actions": [
            "pull-up-account",
            "validate-purchase",
            "record-reason",
            "enter-details",
            "offer-refund",
        ],
        "mandatory_slots": [
            ["username"],
            ["email"],
            ["order_id"],
            ["full_address", "street_address", "account_id"],
        ],
        "agent_action_slots": [
            "reason_slotval",
            "refund_target",
            "details_slotval",
            "amount",
        ],
        "sop_summary": (
            "Customer initiates refund for defective item. Pull up account, validate purchase "
            "with credentials and order ID, record return method, enter customer address or account ID, and offer refund."
        ),
    },
    {
        "flow": "product_defect",
        "intent": "refund_update",
        "canonical_actions": [
            "pull-up-account",
            "validate-purchase",
            "record-reason",
            "offer-refund",
        ],
        "mandatory_slots": [
            ["username"],
            ["email"],
            ["order_id"],
            ["product", "name"],
        ],
        "agent_action_slots": ["reason_slotval", "amount"],
        "sop_summary": (
            "Customer modifies ongoing refund to add defective items. Pull up account, "
            "validate purchase, record additional product item, recalculate and issue updated refund."
        ),
    },
    {
        "flow": "product_defect",
        "intent": "refund_status",
        "canonical_actions": [
            "pull-up-account",
            "validate-purchase",
            "notify-team",
            "update-order",
        ],
        "mandatory_slots": [
            ["username"],
            ["email"],
            ["order_id"],
        ],
        "agent_action_slots": ["company_team", "order_slotval"],
        "sop_summary": (
            "Customer checking status of refund. Pull up account, validate purchase with "
            "credentials and order ID, advise on status or escalate to team/manager."
        ),
    },
    {
        "flow": "product_defect",
        "intent": "return_stain",
        "canonical_actions": [
            "pull-up-account",
            "validate-purchase",
            "membership",
            "enter-details",
            "update-order",
        ],
        "mandatory_slots": [
            ["username"],
            ["email"],
            ["order_id"],
            ["full_address", "street_address"],
        ],
        "agent_action_slots": ["membership_level", "details_slotval", "order_slotval"],
        "sop_summary": (
            "Customer returns item due to stain defect. Pull up account, validate purchase, "
            "verify membership return eligibility, collect shipping return address, and update order for return."
        ),
    },
    {
        "flow": "product_defect",
        "intent": "return_color",
        "canonical_actions": [
            "pull-up-account",
            "validate-purchase",
            "membership",
            "enter-details",
            "update-order",
        ],
        "mandatory_slots": [
            ["username"],
            ["email"],
            ["order_id"],
            ["full_address", "street_address"],
        ],
        "agent_action_slots": ["membership_level", "details_slotval", "order_slotval"],
        "sop_summary": (
            "Customer returns item due to color defect. Pull up account, validate purchase, "
            "verify membership return eligibility, collect shipping return address, and update order for return."
        ),
    },
    {
        "flow": "product_defect",
        "intent": "return_size",
        "canonical_actions": [
            "pull-up-account",
            "validate-purchase",
            "membership",
            "enter-details",
            "update-order",
        ],
        "mandatory_slots": [
            ["username"],
            ["email"],
            ["order_id"],
            ["full_address", "street_address"],
        ],
        "agent_action_slots": ["membership_level", "details_slotval", "order_slotval"],
        "sop_summary": (
            "Customer returns item due to sizing mismatch. Pull up account, validate purchase, "
            "verify membership return eligibility, collect shipping return address, and update order for return."
        ),
    },
    # =========================================================================
    # 5. PURCHASE DISPUTE (8 subflows)
    # =========================================================================
    {
        "flow": "purchase_dispute",
        "intent": "bad_price_competitor",
        "canonical_actions": [
            "pull-up-account",
            "record-reason",
            "verify-identity",
            "promo-code",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": ["reason_slotval"],
        "sop_summary": (
            "Customer disputes purchase price due to lower competitor price. Pull up account, "
            "record dispute reason, verify order identity, and issue courtesy promo code if unresolved."
        ),
    },
    {
        "flow": "purchase_dispute",
        "intent": "bad_price_yesterday",
        "canonical_actions": [
            "pull-up-account",
            "record-reason",
            "verify-identity",
            "promo-code",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": ["reason_slotval"],
        "sop_summary": (
            "Customer disputes price drop yesterday after purchase. Pull up account, "
            "record dispute reason, verify order identity, and issue promo code."
        ),
    },
    {
        "flow": "purchase_dispute",
        "intent": "out_of_stock_general",
        "canonical_actions": ["pull-up-account", "notify-team", "promo-code"],
        "mandatory_slots": [],
        "agent_action_slots": ["company_team"],
        "sop_summary": (
            "Customer inquiring about out-of-stock inventory across catalog. Pull up account, "
            "notify purchasing team, provide courtesy promo code."
        ),
    },
    {
        "flow": "purchase_dispute",
        "intent": "out_of_stock_one_item",
        "canonical_actions": [
            "pull-up-account",
            "record-reason",
            "notify-team",
            "make-purchase",
        ],
        "mandatory_slots": [
            ["product", "name"],
        ],
        "agent_action_slots": ["reason_slotval", "company_team"],
        "sop_summary": (
            "Customer disputes single item being out of stock during purchase. Pull up account, "
            "record specific product, notify merchandising team, and place special backorder."
        ),
    },
    {
        "flow": "purchase_dispute",
        "intent": "promo_code_invalid",
        "canonical_actions": [
            "pull-up-account",
            "ask-the-oracle",
            "membership",
            "promo-code",
        ],
        "mandatory_slots": [],
        "agent_action_slots": ["membership_level"],
        "sop_summary": (
            "Customer attempts to redeem invalid promo code. Pull up account, verify code validity "
            "via oracle, check membership privileges, and issue replacement promo code."
        ),
    },
    {
        "flow": "purchase_dispute",
        "intent": "promo_code_out_of_date",
        "canonical_actions": [
            "pull-up-account",
            "ask-the-oracle",
            "membership",
            "promo-code",
        ],
        "mandatory_slots": [],
        "agent_action_slots": ["membership_level"],
        "sop_summary": (
            "Customer promo code has expired. Pull up account, query oracle, verify member status, "
            "and generate new promo code for eligible tiers."
        ),
    },
    {
        "flow": "purchase_dispute",
        "intent": "mistimed_billing_already_returned",
        "canonical_actions": [
            "pull-up-account",
            "validate-purchase",
            "record-reason",
            "membership",
            "update-order",
        ],
        "mandatory_slots": [
            ["username"],
            ["email"],
            ["order_id"],
        ],
        "agent_action_slots": [
            "reason_slotval",
            "membership_level",
            "order_slotval",
        ],
        "sop_summary": (
            "Customer billed for item already returned. Pull up account, validate purchase "
            "with credentials, verify return timeframe and membership, process credit via update-order."
        ),
    },
    {
        "flow": "purchase_dispute",
        "intent": "mistimed_billing_never_bought",
        "canonical_actions": [
            "pull-up-account",
            "validate-purchase",
            "ask-the-oracle",
            "membership",
            "update-order",
        ],
        "mandatory_slots": [
            ["username"],
            ["email"],
            ["order_id"],
        ],
        "agent_action_slots": ["membership_level", "order_slotval"],
        "sop_summary": (
            "Customer billed for unpurchased item. Pull up account, validate order records, "
            "check oracle, verify membership tier, issue refund/credit via update-order."
        ),
    },
    # =========================================================================
    # 6. SHIPPING ISSUE (4 subflows)
    # =========================================================================
    {
        "flow": "shipping_issue",
        "intent": "status",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "validate-purchase",
            "ask-the-oracle",
            "update-order",
        ],
        "mandatory_slots": [
            ["order_id"],
            ["username"],
            ["email"],
        ],
        "agent_action_slots": ["order_slotval"],
        "sop_summary": (
            "Customer inquiring about shipment status. Pull up account, verify identity and order "
            "purchase details, consult oracle, update order delivery details if required."
        ),
    },
    {
        "flow": "shipping_issue",
        "intent": "manage",
        "canonical_actions": [
            "pull-up-account",
            "shipping-status",
            "validate-purchase",
            "update-order",
        ],
        "mandatory_slots": [
            ["username"],
            ["email"],
            ["order_id"],
        ],
        "agent_action_slots": ["shipping_option", "order_slotval"],
        "sop_summary": (
            "Customer modifying in-flight shipment (reroute/reschedule). Pull up account, "
            "check shipping status, validate purchase credentials, update order destination/timing."
        ),
    },
    {
        "flow": "shipping_issue",
        "intent": "missing",
        "canonical_actions": [
            "pull-up-account",
            "validate-purchase",
            "record-reason",
            "update-order",
            "make-purchase",
        ],
        "mandatory_slots": [
            ["username"],
            ["email"],
            ["order_id"],
            ["full_address", "street_address"],
            ["product", "name"],
        ],
        "agent_action_slots": ["reason_slotval", "order_slotval"],
        "sop_summary": (
            "Package marked delivered but missing. Pull up account, validate purchase, "
            "record delivery wait time, gather shipping address and item details to reorder replacement."
        ),
    },
    {
        "flow": "shipping_issue",
        "intent": "cost",
        "canonical_actions": [
            "pull-up-account",
            "validate-purchase",
            "shipping-status",
            "update-order",
            "offer-refund",
        ],
        "mandatory_slots": [
            ["username"],
            ["email"],
            ["order_id"],
        ],
        "agent_action_slots": ["shipping_option", "order_slotval", "amount"],
        "sop_summary": (
            "Customer disputes unexpected shipping charges. Pull up account, validate purchase, "
            "verify shipping progress, waive fee or offer refund."
        ),
    },
    # =========================================================================
    # 7. SINGLE ITEM QUERY (4 subflows)
    # =========================================================================
    {
        "flow": "single_item_query",
        "intent": "boots",
        "canonical_actions": ["search-faq", "search-boots", "select-faq"],
        "mandatory_slots": [],
        "agent_action_slots": ["single_item_query"],
        "sop_summary": (
            "Customer inquiring about boots product details (material, care, sizing). "
            "Search FAQ database, navigate to boots category, select relevant policy response."
        ),
    },
    {
        "flow": "single_item_query",
        "intent": "shirt",
        "canonical_actions": ["search-faq", "search-shirt", "select-faq"],
        "mandatory_slots": [],
        "agent_action_slots": ["single_item_query"],
        "sop_summary": (
            "Customer inquiring about shirt product details. Search FAQ database, "
            "navigate to shirt category, select relevant policy response."
        ),
    },
    {
        "flow": "single_item_query",
        "intent": "jeans",
        "canonical_actions": ["search-faq", "search-jeans", "select-faq"],
        "mandatory_slots": [],
        "agent_action_slots": ["single_item_query"],
        "sop_summary": (
            "Customer inquiring about jeans product details. Search FAQ database, "
            "navigate to jeans category, select relevant policy response."
        ),
    },
    {
        "flow": "single_item_query",
        "intent": "jacket",
        "canonical_actions": ["search-faq", "search-jacket", "select-faq"],
        "mandatory_slots": [],
        "agent_action_slots": ["single_item_query"],
        "sop_summary": (
            "Customer inquiring about jacket product details. Search FAQ database, "
            "navigate to jacket category, select relevant policy response."
        ),
    },
    # =========================================================================
    # 8. STOREWIDE QUERY (4 subflows)
    # =========================================================================
    {
        "flow": "storewide_query",
        "intent": "pricing",
        "canonical_actions": ["search-faq", "search-pricing", "select-faq"],
        "mandatory_slots": [],
        "agent_action_slots": ["storewide_query"],
        "sop_summary": (
            "Customer asking general pricing / discount questions. Search FAQ repository, "
            "navigate to pricing, and provide canonical answer."
        ),
    },
    {
        "flow": "storewide_query",
        "intent": "membership",
        "canonical_actions": ["search-faq", "search-membership", "select-faq"],
        "mandatory_slots": [],
        "agent_action_slots": ["storewide_query"],
        "sop_summary": (
            "Customer asking storewide membership tier questions. Search FAQ repository, "
            "navigate to membership, and provide canonical answer."
        ),
    },
    {
        "flow": "storewide_query",
        "intent": "timing",
        "canonical_actions": ["search-faq", "search-timing", "select-faq"],
        "mandatory_slots": [],
        "agent_action_slots": ["storewide_query"],
        "sop_summary": (
            "Customer asking operating hours / delivery timing questions. Search FAQ repository, "
            "navigate to timing, and provide canonical answer."
        ),
    },
    {
        "flow": "storewide_query",
        "intent": "policy",
        "canonical_actions": ["search-faq", "search-policy", "select-faq"],
        "mandatory_slots": [],
        "agent_action_slots": ["storewide_query"],
        "sop_summary": (
            "Customer asking general storewide return / warranty policy questions. "
            "Search FAQ repository, navigate to policy, and provide canonical answer."
        ),
    },
    # =========================================================================
    # 9. SUBSCRIPTION INQUIRY (6 subflows)
    # =========================================================================
    {
        "flow": "subscription_inquiry",
        "intent": "status_active",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "subscription-status",
            "send-link",
            "enter-details",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": ["details_slotval"],
        "sop_summary": (
            "Customer checking active subscription status. Pull up account, verify identity "
            "and order, query subscription system, and send portal link."
        ),
    },
    {
        "flow": "subscription_inquiry",
        "intent": "status_due_amount",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "subscription-status",
            "send-link",
            "enter-details",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": ["details_slotval"],
        "sop_summary": (
            "Customer checking outstanding subscription amount. Pull up account, verify identity "
            "and order, query subscription system, and send portal link."
        ),
    },
    {
        "flow": "subscription_inquiry",
        "intent": "status_due_date",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "subscription-status",
            "send-link",
            "enter-details",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": ["details_slotval"],
        "sop_summary": (
            "Customer checking subscription renewal date. Pull up account, verify identity "
            "and order, query subscription system, and send portal link."
        ),
    },
    {
        "flow": "subscription_inquiry",
        "intent": "manage_pay_bill",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "subscription-status",
            "enter-details",
            "update-account",
        ],
        "mandatory_slots": [
            ["order_id"],
            ["payment_method", "amount"],
        ],
        "agent_action_slots": ["details_slotval", "account_slotval"],
        "sop_summary": (
            "Customer paying periodic subscription charge. Pull up account, verify identity, "
            "confirm billing amount, record payment details, renew subscription via update-account."
        ),
    },
    {
        "flow": "subscription_inquiry",
        "intent": "manage_extension",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "membership",
            "update-account",
            "enter-details",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": [
            "membership_level",
            "account_slotval",
            "details_slotval",
        ],
        "sop_summary": (
            "Customer requesting billing extension. Pull up account, verify identity, "
            "assess membership privileges, grant extension via update-account or escalate."
        ),
    },
    {
        "flow": "subscription_inquiry",
        "intent": "manage_dispute_bill",
        "canonical_actions": [
            "pull-up-account",
            "verify-identity",
            "membership",
            "ask-the-oracle",
            "offer-refund",
        ],
        "mandatory_slots": [
            ["order_id"],
        ],
        "agent_action_slots": ["membership_level", "amount"],
        "sop_summary": (
            "Customer disputes recurring subscription fee charge. Pull up account, verify identity, "
            "check membership level and oracle, calculate discrepancy, and issue refund."
        ),
    },
    # =========================================================================
    # 10. TROUBLESHOOT SITE (4 subflows)
    # =========================================================================
    {
        "flow": "troubleshoot_site",
        "intent": "credit_card",
        "canonical_actions": [
            "try-again",
            "log-out-in",
            "enter-details",
            "make-purchase",
        ],
        "mandatory_slots": [
            ["product", "name"],
        ],
        "agent_action_slots": ["details_slotval"],
        "sop_summary": (
            "Customer experiencing payment processing errors on website. Advise customer to retry "
            "and re-authenticate, or agent enters troubleshoot code and manually places order."
        ),
    },
    {
        "flow": "troubleshoot_site",
        "intent": "shopping_cart",
        "canonical_actions": [
            "try-again",
            "log-out-in",
            "enter-details",
            "make-purchase",
        ],
        "mandatory_slots": [
            ["product", "name"],
        ],
        "agent_action_slots": ["details_slotval"],
        "sop_summary": (
            "Customer shopping cart not updating or persisting. Guide through cache refresh "
            "and re-login, or agent places order directly."
        ),
    },
    {
        "flow": "troubleshoot_site",
        "intent": "search_results",
        "canonical_actions": [
            "try-again",
            "log-out-in",
            "make-purchase",
            "instructions",
            "notify-team",
        ],
        "mandatory_slots": [],
        "agent_action_slots": ["company_team"],
        "sop_summary": (
            "Customer search bar failing or empty results. Provide browser instruction to clear "
            "cookies, report bug to website team, or assist with direct purchase."
        ),
    },
    {
        "flow": "troubleshoot_site",
        "intent": "slow_speed",
        "canonical_actions": [
            "try-again",
            "log-out-in",
            "make-purchase",
            "instructions",
            "notify-team",
        ],
        "mandatory_slots": [],
        "agent_action_slots": ["company_team"],
        "sop_summary": (
            "Customer experiencing high page latency. Guide through browser hygiene, "
            "file performance incident with website team, or place order on customer behalf."
        ),
    },
]


def build_intent_schema(actions_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Construct the normalized intent schema DataFrame for all 55 intents.

    Parameters
    ----------
    actions_df : pd.DataFrame, optional
        Preloaded actions DataFrame containing action_name and required_slots.
        If None, reads from ACTIONS_PARQUET.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - flow: str
        - intent: str
        - canonical_actions: list[str]
        - mandatory_slots: list[list[str]] (CNF: outer list AND, inner list OR)
        - num_mandatory_slots: int
        - agent_action_slots: list[str]
        - all_slots: list[str] (union of slots from all canonical actions)
        - sop_summary: str
    """
    if actions_df is None:
        from abcd_helper.config import ACTIONS_PARQUET

        actions_df = pd.read_parquet(ACTIONS_PARQUET)

    known_actions = set(actions_df["action_name"])
    known_slots = set()
    for s_list in actions_df["required_slots"]:
        known_slots.update(s_list)
    # Add 'name' as a valid alias for merchandise 'product' from ABCD utterance slot_tags
    known_slots.add("name")

    action_to_slots = dict(zip(actions_df["action_name"], actions_df["required_slots"]))

    rows = []
    for entry in INTENT_SCHEMA_DATA:
        flow = entry["flow"]
        intent = entry["intent"]
        canonical_actions = entry["canonical_actions"]
        mandatory_slots = entry["mandatory_slots"]
        agent_action_slots = entry["agent_action_slots"]
        sop_summary = entry["sop_summary"]

        # ---------------------------------------------------------------------
        # Validation: Verify canonical actions against official action ontology
        # ---------------------------------------------------------------------
        for act in canonical_actions:
            if act not in known_actions:
                raise ValueError(
                    f"Unknown action '{act}' in intent '{intent}' canonical_actions. "
                    f"Must match official ontology actions."
                )

        # ---------------------------------------------------------------------
        # Validation: Verify mandatory slots (CNF) against actions ontology
        # ---------------------------------------------------------------------
        if not isinstance(mandatory_slots, list):
            raise TypeError(
                f"mandatory_slots for intent '{intent}' must be a list of lists."
            )
        for clause in mandatory_slots:
            if not isinstance(clause, list) or len(clause) == 0:
                raise ValueError(
                    f"Each CNF clause in intent '{intent}' mandatory_slots must be a non-empty list."
                )
            for slot in clause:
                if slot not in known_slots:
                    raise ValueError(
                        f"Unknown slot '{slot}' in intent '{intent}' mandatory_slots. "
                        f"Must be defined in the ABCD actions ontology."
                    )

        # ---------------------------------------------------------------------
        # Validation: Verify agent action slots against actions ontology
        # ---------------------------------------------------------------------
        if not isinstance(agent_action_slots, list):
            raise TypeError(f"agent_action_slots for intent '{intent}' must be a list.")
        for slot in agent_action_slots:
            if slot not in known_slots:
                raise ValueError(
                    f"Unknown slot '{slot}' in intent '{intent}' agent_action_slots. "
                    f"Must be defined in the ABCD actions ontology."
                )

        # Compute all_slots as the unique union of slots from all canonical actions in order
        all_slots_list = []
        for act in canonical_actions:
            slots = action_to_slots.get(act, [])
            for s in slots:
                if s not in all_slots_list:
                    all_slots_list.append(s)
                if s == "product" and "name" not in all_slots_list:
                    all_slots_list.append("name")

        rows.append(
            {
                "flow": flow,
                "intent": intent,
                "canonical_actions": canonical_actions,
                "mandatory_slots": mandatory_slots,
                "num_mandatory_slots": len(mandatory_slots),
                "agent_action_slots": agent_action_slots,
                "all_slots": all_slots_list,
                "sop_summary": sop_summary,
            }
        )

    df = pd.DataFrame(rows)
    return ensure_native_lists(df)


def ensure_native_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all nested list columns in intent schema are pure native Python lists (not numpy arrays)."""
    if "mandatory_slots" in df.columns:
        df["mandatory_slots"] = df["mandatory_slots"].apply(
            lambda val: [
                [str(x) for x in clause]
                if hasattr(clause, "__iter__") and not isinstance(clause, (str, bytes))
                else [str(clause)]
                for clause in val
            ]
            if hasattr(val, "__iter__") and not isinstance(val, (str, bytes))
            else []
        )
    for col in ["canonical_actions", "agent_action_slots", "all_slots"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda val: [str(x) for x in val]
                if hasattr(val, "__iter__") and not isinstance(val, (str, bytes))
                else []
            )
    return df


def save_intent_schema(
    df: pd.DataFrame | None = None, dest_path=INTENT_SCHEMA_PARQUET
) -> pd.DataFrame:
    """Save intent schema to Parquet format with native python list columns."""
    if df is None:
        df = build_intent_schema()
    df = ensure_native_lists(df)
    df.to_parquet(dest_path, index=False)
    return df

