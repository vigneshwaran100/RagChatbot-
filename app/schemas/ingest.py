from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str = Field(..., description="UUID assigned to the ingested document")
    filename: str
    file_type: str
    total_chunks: int
    collection: str
    message: str = "Document ingested successfully"


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    total_chunks: int
    ingested_at: str


class ListDocumentsResponse(BaseModel):
    collection: str
    total_documents: int
    documents: list[DocumentInfo]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted_chunks: int
    message: str = "Document deleted successfully"
