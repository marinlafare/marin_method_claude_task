from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from operations.library import list_melody_assets, list_melody_index, load_melody_asset_path, load_melody_json
from operations.schemas import MelodyAssetItem, MelodyAssetsResponse, MelodyIndexItem, MelodyIndexResponse, MelodyJsonResponse

router = APIRouter()


@router.get("/melodies", response_model=MelodyIndexResponse)
def list_melodies() -> MelodyIndexResponse:
    core_items = list_melody_index()
    items = [MelodyIndexItem(id=i.id, label=i.label, json_path=i.json_path) for i in core_items]
    return MelodyIndexResponse(ok=True, items=items)


@router.get("/melodies/{melody_id}", response_model=MelodyJsonResponse)
def get_melody_json(melody_id: str) -> MelodyJsonResponse:
    try:
        core = load_melody_json(melody_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Melody not found: {melody_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return MelodyJsonResponse(ok=True, id=core.id, json_path=core.json_path, data=core.data)


@router.get("/melodies/{melody_id}/assets", response_model=MelodyAssetsResponse)
def list_assets(melody_id: str) -> MelodyAssetsResponse:
    try:
        assets = list_melody_assets(melody_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Melody not found: {melody_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    items = [
        MelodyAssetItem(
            kind=a.kind,
            filename=a.filename,
            url=f"/pitch-guide/melodies/{melody_id}/assets/{a.kind}",
        )
        for a in assets
    ]
    return MelodyAssetsResponse(ok=True, id=melody_id, assets=items)


@router.get("/melodies/{melody_id}/assets/{kind}")
def get_asset(melody_id: str, kind: str) -> FileResponse:
    try:
        p = load_melody_asset_path(melody_id, kind)  # type: ignore[arg-type]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Asset not found: {melody_id}/{kind}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return FileResponse(path=str(p), media_type="audio/wav", filename=p.name)
