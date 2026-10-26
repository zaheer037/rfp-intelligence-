"""Google ADK A2A transport over the same application agent facade."""

from rfp.agents.agent import RfpWorkflow


def create_a2a_app(settings, service):
    """Build discovery and RPC routes; parent middleware owns their authentication."""
    from a2a.client.card_resolver import parse_agent_card
    from google.adk.a2a.utils.agent_to_a2a import to_a2a

    card = parse_agent_card(
        {
            "name": "RFP Intelligence",
            "description": "Index, extract and answer questions over RFP bids",
            "url": settings.public_base_url.rstrip("/") + "/a2a/",
            "version": "0.1.0",
            "capabilities": {"streaming": False},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["application/json"],
            "skills": [
                {
                    "id": "rfp",
                    "name": "RFP workflows",
                    "description": "Cited extraction and Q&A",
                    "tags": ["rfp", "search"],
                }
            ],
            "securitySchemes": {"apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}},
            "security": [{"apiKey": []}],
        }
    )
    return to_a2a(
        RfpWorkflow(
            name="rfp_orchestrator",
            service=service,
            description="Index, extract and answer questions over RFP bids",
        ),
        agent_card=card,
    )
