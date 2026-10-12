from pydantic import BaseModel, Field


class ReformulatedQuery(BaseModel):
    """
    Structured output from the LLM query reformulation step.
    This schema is passed to the LLM as a JSON format instruction
    so that the output can be parsed reliably.
    """

    refined_query: str = Field(
        ...,
        description=(
            "A semantically cleaner, complete rewrite of the user question "
            "suitable for dense vector search. Remove ambiguity and resolve "
            "pronouns against prior context if available."
        ),
    )
    search_terms: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "Key noun phrases and technical terms extracted from the question "
            "that should appear in relevant documents."
        ),
    )
    intent: str = Field(
        ...,
        description=(
            "One-sentence description of what the user is trying to learn or do."
        ),
    )
    requires_multiple_chunks: bool = Field(
        default=False,
        description=(
            "True when answering requires aggregating information from "
            "multiple document sections."
        ),
    )
