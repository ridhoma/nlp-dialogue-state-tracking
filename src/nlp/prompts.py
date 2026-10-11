SLOTS_EXTRACTION_PROMPT = """
# The Task
You are an AI annotator tasked with dialogue slots annotation. You are given conversation dataset containing many conversations. In each conversation, you can see exchanges between AGENT and CUSTOMER. Your job is to tag slots for CUSTOMER's turn. Only flag the CUSTOMER's turn, but you can use AGENT's turn as context to infer the information provided in CUSTOMER's turn. The list of available slots and their descriptions are provided below. You should only chose from this list.
```
{ 
   "account_id": "The customer enters their account ID.",
   "amount": "The customer specifies a dollar amount or price.",
   "email": "The customer enters their email address.",
   "name": "The customer specifies a merchandise product name.",
   "order_id": "The customer specifies an order ID number.",
   "payment_method": "The customer specifies a payment method.",
   "phone": "The customer provides a telephone phone number.",
   "pin_number": "The customer enters their PIN number.",
   "street_address": "The customer specifies a street address.",
   "username": "The customer writes their account username in the message.",
   "zip_code": "The customer provides a postal zip code.",
}
```

# The Data
The conversation data is a json file with `<conversation_id>` as the key, and list of turn exchanges as the value. Below is example of one conversation.
```
{
    <conversation_id>: [
        {
            turn_id: 1,
            speaker: "AGENT",
            text: "Hello and thank you for reaching out to us! What can I help you with today?"
        },
        {
            turn_id: 2, 
            speaker: "CUSTOMER", 
            text: "..."
        },
        ...
    ] 
}
```

Your task is to go through each CUSTOMER turn and try to identify the slots that are being provided by the customer based on the description above. Return the output in a separate json array file wth following format:
```
[ 
    {
        "conversation_id": "2101",
        "turn_id": 2,
        "slots": ["account_id", "email", "username"] # multiple slots
    },
    {
        "conversation_id": "2101",
        "turn_id": 4,
        "slots": [] # no slots
    },
    {
        "conversation_id": "2101",
        "turn_id": 6,
        "slots": ["address"] # single slot
    }
]
```
Notice that output file only contains CUSTOMER's turns. AGENT's turns are skipped, but can be used as context for your slot inference.
"""