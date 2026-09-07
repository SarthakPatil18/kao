"""
Stage 3 — Web Discovery

Performs real, live reverse-image search via SerpApi (Google Lens engine)
with a Bing Visual Search fallback.

Downloads candidate images under strict security limits:
  - MIME allowlist (image/jpeg, image/png, image/webp)
  - Max file size (5 MB default)
  - Connection timeout
  - No execution of fetched content
"""

import io
import re
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

import requests
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_MAX_SIZE_MB = 5
DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_BASE = 2.0

# Platform classification patterns
PLATFORM_PATTERNS = {
    "linkedin": re.compile(r"linkedin\.com", re.I),
    "github": re.compile(r"github\.com", re.I),
    "twitter": re.compile(r"(twitter\.com|x\.com)", re.I),
    "instagram": re.compile(r"instagram\.com", re.I),
    "facebook": re.compile(r"facebook\.com", re.I),
    "youtube": re.compile(r"youtube\.com", re.I),
    "reddit": re.compile(r"reddit\.com", re.I),
    "medium": re.compile(r"medium\.com", re.I),
    "tiktok": re.compile(r"tiktok\.com", re.I),
    "threads": re.compile(r"threads\.net", re.I),
    "bluesky": re.compile(r"(bsky\.app|bluesky)", re.I),
    "substack": re.compile(r"substack\.com", re.I),
    "pinterest": re.compile(r"pinterest\.com", re.I),
    "news": re.compile(
        r"(reuters|apnews|bbc|cnn|nytimes|theguardian|washingtonpost|bloomberg|forbes|techcrunch)"
        r"\.com",
        re.I,
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single result from reverse-image search."""
    title: str
    url: str
    thumbnail_url: str = ""
    source: str = ""                    # e.g. "Google Lens", "Bing Visual"
    platform: str = "web"               # Classified platform
    snippet: str = ""                   # Surrounding text / caption
    position: int = 0                   # Rank in results
    image_bytes: Optional[bytes] = None # Downloaded candidate image


# ---------------------------------------------------------------------------
# Platform classification
# ---------------------------------------------------------------------------

def classify_platform(url: str) -> str:
    """Classify a URL into a known platform category or clean domain name."""
    if not url:
        return "web"
    for platform, pattern in PLATFORM_PATTERNS.items():
        if pattern.search(url):
            return platform
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        parts = netloc.split(".")
        if len(parts) >= 2:
            return parts[-2]
        return netloc or "web"
    except Exception:
        return "web"


# ---------------------------------------------------------------------------
# SerpApi — Google Lens
# ---------------------------------------------------------------------------

def search_serpapi(
    image_path: str,
    api_key: str,
    num_results: int = 10,
) -> List[SearchResult]:
    """Search via SerpApi Google Lens engine.

    Args:
        image_path: Path to the image file to search.
        api_key: SerpApi API key.
        num_results: Maximum results to return.

    Returns:
        List of SearchResult objects.
    """
    try:
        from serpapi import GoogleSearch
    except ImportError:
        raise ImportError(
            "google-search-results not installed. Run: pip install google-search-results"
        )

    params = {
        "engine": "google_lens",
        "url": None,   # Will use uploaded image
        "api_key": api_key,
    }

    # Upload image to SerpApi
    # SerpApi Google Lens supports image_url or direct file upload
    # For local files, we need to use the image_url approach via their upload
    # or use the Google reverse image search engine as fallback
    
    # Try Google Reverse Image Search first (more reliable for face matching)
    params_reverse = {
        "engine": "google_reverse_image",
        "image_url": None,
        "api_key": api_key,
    }

    # Read and encode image for upload
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Use SerpApi's Google Lens with image upload
    import tempfile
    import base64

    search = GoogleSearch({
        "engine": "google_lens",
        "api_key": api_key,
        "image": image_path,
    })
    
    try:
        raw_results = search.get_dict()
    except Exception as e:
        logger.warning(f"SerpApi Google Lens failed: {e}")
        # Fallback: try reverse image search
        try:
            search = GoogleSearch({
                "engine": "google_reverse_image",
                "api_key": api_key,
                "image": image_path,
            })
            raw_results = search.get_dict()
        except Exception as e2:
            logger.error(f"SerpApi reverse image also failed: {e2}")
            return []

    results: List[SearchResult] = []

    # Parse visual_matches (Google Lens response format)
    visual_matches = raw_results.get("visual_matches", [])
    for i, match in enumerate(visual_matches[:num_results]):
        url = match.get("link", "")
        sr = SearchResult(
            title=match.get("title", ""),
            url=url,
            thumbnail_url=match.get("thumbnail", ""),
            source="Google Lens",
            platform=classify_platform(url),
            snippet=match.get("snippet", match.get("source", "")),
            position=i + 1,
        )
        results.append(sr)

    # Also check inline_images or image_results
    for key in ("image_results", "inline_images"):
        for i, img in enumerate(raw_results.get(key, [])):
            url = img.get("link", img.get("source", ""))
            if url and not any(r.url == url for r in results):
                sr = SearchResult(
                    title=img.get("title", ""),
                    url=url,
                    thumbnail_url=img.get("thumbnail", img.get("original", "")),
                    source="Google Lens",
                    platform=classify_platform(url),
                    snippet=img.get("snippet", ""),
                    position=len(results) + 1,
                )
                results.append(sr)
                if len(results) >= num_results:
                    break

    logger.info(f"SerpApi returned {len(results)} results")
    return results


# ---------------------------------------------------------------------------
# Bing Visual Search (fallback)
# ---------------------------------------------------------------------------

def search_bing_visual(
    image_path: str,
    api_key: str,
    num_results: int = 10,
) -> List[SearchResult]:
    """Search via Bing Visual Search API (fallback).

    Args:
        image_path: Path to the image file.
        api_key: Bing Search API key.
        num_results: Maximum results to return.

    Returns:
        List of SearchResult objects.
    """
    endpoint = "https://api.bing.microsoft.com/v7.0/images/visualsearch"

    with open(image_path, "rb") as f:
        image_data = f.read()

    headers = {"Ocp-Apim-Subscription-Key": api_key}
    files = {"image": ("image.jpg", image_data, "image/jpeg")}

    try:
        resp = requests.post(
            endpoint, headers=headers, files=files, timeout=DEFAULT_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Bing Visual Search failed: {e}")
        return []

    results: List[SearchResult] = []
    tags = data.get("tags", [])
    for tag in tags:
        for action in tag.get("actions", []):
            if action.get("actionType") in (
                "PagesIncluding",
                "VisualSearch",
            ):
                for i, item in enumerate(action.get("data", {}).get("value", [])):
                    url = item.get("hostPageUrl", item.get("contentUrl", ""))
                    sr = SearchResult(
                        title=item.get("name", ""),
                        url=url,
                        thumbnail_url=item.get("thumbnailUrl", ""),
                        source="Bing Visual",
                        platform=classify_platform(url),
                        snippet=item.get("snippet", ""),
                        position=len(results) + 1,
                    )
                    results.append(sr)
                    if len(results) >= num_results:
                        break

    logger.info(f"Bing Visual returned {len(results)} results")
    return results


# ---------------------------------------------------------------------------
# Unified search
# ---------------------------------------------------------------------------

def search_by_image(
    image_path: str,
    serpapi_key: Optional[str] = None,
    bing_key: Optional[str] = None,
    num_results: int = 10,
    allow_simulated: bool = False,
) -> List[SearchResult]:
    """Run reverse-image search, trying SerpApi first, then Bing fallback.

    If no keys are provided and allow_simulated is True, returns realistic
    multi-platform simulated candidates with embedded candidate image bytes.
    """
    results: List[SearchResult] = []

    if serpapi_key:
        try:
            results = search_serpapi(image_path, serpapi_key, num_results)
        except Exception as e:
            logger.warning(f"SerpApi failed, trying Bing fallback: {e}")

    if not results and bing_key:
        results = search_bing_visual(image_path, bing_key, num_results)

    if not results and allow_simulated:
        logger.info("Using simulated search discovery (offline / demo mode).")
        results = generate_simulated_candidates(image_path, num_results)

    if not results:
        logger.warning("No results from any search provider.")

    return results


def generate_simulated_candidates(
    image_path: str,
    num_results: int = 4,
) -> List[SearchResult]:
    """Generate realistic simulated candidates for offline or demo testing.

    Derives candidate images from the input image to produce a realistic
    multi-tier score distribution (HIGH, MEDIUM, and LOW candidates).
    """
    candidates: List[SearchResult] = []
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        import cv2

        src_img = Image.open(image_path).convert("RGB")
        w, h = src_img.size

        # Candidate 1: High match (LinkedIn profile avatar)
        # Resize slightly and compress
        c1_buf = io.BytesIO()
        c1_img = src_img.resize((min(w, 512), min(h, 512)))
        c1_img.save(c1_buf, format="JPEG", quality=88)
        c1_bytes = c1_buf.getvalue()

        candidates.append(SearchResult(
            title="Public Professional Profile — Staff Systems Architect",
            url="https://linkedin.com/in/verified-subject-profile",
            thumbnail_url="",
            source="Simulated Search (Demo)",
            platform="linkedin",
            snippet="Staff distributed systems engineer and cryptography researcher. Focus on decentralized privacy architectures, Merkle attestation, and verifiable compute.",
            position=1,
            image_bytes=c1_bytes,
        ))

        # Candidate 2: Medium-High match (GitHub profile avatar)
        # Slight brightness shift and border crop
        c2_buf = io.BytesIO()
        enhancer = ImageEnhance.Brightness(src_img)
        c2_img = enhancer.enhance(1.08)
        c2_img = c2_img.resize((min(w, 400), min(h, 400)))
        c2_img.save(c2_buf, format="JPEG", quality=80)
        c2_bytes = c2_buf.getvalue()

        candidates.append(SearchResult(
            title="GitHub Developer Profile — Identity & Attestation Core",
            url="https://github.com/subject-identity-dev",
            thumbnail_url="",
            source="Simulated Search (Demo)",
            platform="github",
            snippet="Core contributor to decentralized identity standards, attestation frameworks, and biometric privacy protocols. Verified open-source committer.",
            position=2,
            image_bytes=c2_bytes,
        ))

        # Candidate 3: Medium match (Tech Publication Article)
        c3_buf = io.BytesIO()
        c3_img = src_img.filter(ImageFilter.GaussianBlur(radius=0.6))
        c3_img = ImageEnhance.Contrast(c3_img).enhance(1.15)
        c3_img.save(c3_buf, format="JPEG", quality=75)
        c3_bytes = c3_buf.getvalue()

        candidates.append(SearchResult(
            title="Medium Publication: Engineering Tamper-Evident Attestations in Web3",
            url="https://medium.com/@subject/tamper-evident-pipelines-2026",
            thumbnail_url="",
            source="Simulated Search (Demo)",
            platform="medium",
            snippet="Discussion on privacy-preserving biometric discovery without raw embedding persistence on public blockchains. Practical Merkle tree architectures.",
            position=3,
            image_bytes=c3_bytes,
        ))

        # Candidate 4: Low match (Unrelated Tech Conference Panel)
        # Generate an altered image that will have low face similarity
        c4_buf = io.BytesIO()
        # Flip horizontal and heavy color tint to simulate different subject
        c4_img = src_img.transpose(Image.FLIP_LEFT_RIGHT)
        c4_img = ImageEnhance.Color(c4_img).enhance(0.2)
        c4_img = c4_img.resize((200, 200))
        c4_img.save(c4_buf, format="JPEG", quality=50)
        c4_bytes = c4_buf.getvalue()

        candidates.append(SearchResult(
            title="Global Technology Summit 2026 Speaker Lineup",
            url="https://technews.com/summit-2026/coverage",
            thumbnail_url="",
            source="Simulated Search (Demo)",
            platform="news",
            snippet="Coverage of the annual global technology keynote addressing cloud computing, serverless infrastructure, and enterprise data streaming.",
            position=4,
            image_bytes=c4_bytes,
        ))

    except Exception as e:
        logger.error(f"Error generating simulated candidates: {e}")

    return candidates[:num_results]


# ---------------------------------------------------------------------------
# Candidate image download
# ---------------------------------------------------------------------------

def download_candidate(
    url: str,
    max_size_mb: float = DEFAULT_MAX_SIZE_MB,
    timeout: int = DEFAULT_TIMEOUT,
    fallback_bytes: Optional[bytes] = None,
) -> Optional[bytes]:
    """Download a candidate image with strict security limits.

    Args:
        url: URL of the image to download.
        max_size_mb: Maximum file size in MB.
        timeout: Request timeout in seconds.
        fallback_bytes: Optional pre-loaded bytes (e.g. for simulated candidates).

    Returns:
        Image bytes if valid, None otherwise.
    """
    if fallback_bytes is not None and len(fallback_bytes) > 0:
        return fallback_bytes

    if not url or not url.startswith(("http://", "https://")):
        return None

    max_bytes = int(max_size_mb * 1024 * 1024)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                stream=True,
                headers={"User-Agent": "Provenance-Bot/1.0 (evidence-verification)"},
            )

            # Check MIME type
            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
            if content_type not in ALLOWED_MIME_TYPES:
                logger.warning(
                    f"Rejected {url}: MIME type {content_type} not in allowlist"
                )
                return None

            # Check content length if available
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                logger.warning(f"Rejected {url}: size {content_length} exceeds limit")
                return None

            # Stream download with size check
            chunks = []
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    logger.warning(f"Rejected {url}: exceeded size limit during download")
                    return None
                chunks.append(chunk)

            image_bytes = b"".join(chunks)

            # Validate it's actually a decodable image
            try:
                img = Image.open(io.BytesIO(image_bytes))
                img.verify()
            except Exception:
                logger.warning(f"Rejected {url}: failed image verification")
                return None

            return image_bytes

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = BACKOFF_BASE ** attempt
                logger.info(f"Rate limited on {url}, retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            logger.warning(f"HTTP error downloading {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error downloading {url}: {e}")
            return None

    return None
