from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
import re

SAFE_URL_RE = re.compile(r'^https?://', re.I)

class SourceItem(BaseModel):
    url: HttpUrl
    sha256: Optional[str] = None
    filename: Optional[str] = None
    license: Optional[str] = None

class Manifest(BaseModel):
    name: str
    sources: List[SourceItem] = Field(min_length=1)
