from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class ModelSpecsResult:
    input_brand: str
    input_model: str

    exact_model_confirmed: bool = False
    confirmed_model: str | None = None
    suggested_model: str | None = None
    match_notes: str | None = None

    appliance_type: str | None = None

    size_value: float | None = None
    size_unit: str | None = None
    color: str | None = None

    notes_block: str | None = None

    source_name: str | None = None
    source_url: str | None = None

    confidence: str = "low"

    sources: list[dict[str, str]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelSpecsService:
    """
    Search appliance specifications using OpenAI Responses API
    with web search.

    IMPORTANT:
    - This service only researches and returns data.
    - It does NOT write to the WCCR database.
    - It does NOT modify ApplianceUnit.
    - It does NOT modify Receiving.
    """

    MODEL = "gpt-5.5"

    VALID_CONFIDENCE = {
        "high",
        "medium",
        "low",
    }

    def __init__(
        self,
        *,
        api_key: str | None = None,
    ):
        load_dotenv(override=True)

        key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )

        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=key
        )

    @staticmethod
    def _clean_text(
        value: Any,
        *,
        max_length: int | None = None,
    ) -> str | None:

        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        if max_length is not None:
            text = text[:max_length]

        return text

    @staticmethod
    def _clean_float(
        value: Any,
    ) -> float | None:

        if value in (
            None,
            "",
        ):
            return None

        try:
            number = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

        if number < 0:
            return None

        return number

    @classmethod
    def _normalize_result(
        cls,
        *,
        brand: str,
        model: str,
        payload: dict[str, Any],
    ) -> ModelSpecsResult:

        confidence = (
            cls._clean_text(
                payload.get("confidence")
            )
            or "low"
        ).lower()

        if confidence not in cls.VALID_CONFIDENCE:
            confidence = "low"

        raw_sources = payload.get(
            "sources"
        )

        sources: list[dict[str, str]] = []

        if isinstance(
            raw_sources,
            list,
        ):
            for item in raw_sources:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                name = cls._clean_text(
                    item.get("name"),
                    max_length=200,
                )

                url = cls._clean_text(
                    item.get("url"),
                    max_length=2000,
                )

                if not name and not url:
                    continue

                sources.append(
                    {
                        "name": name or "",
                        "url": url or "",
                    }
                )

        return ModelSpecsResult(
            input_brand=brand,
            input_model=model,

            exact_model_confirmed=bool(
                payload.get(
                    "exact_model_confirmed",
                    False,
                )
            ),

            confirmed_model=cls._clean_text(
                payload.get(
                    "confirmed_model"
                ),
                max_length=120,
            ),

            suggested_model=cls._clean_text(
                payload.get(
                    "suggested_model"
                ),
                max_length=120,
            ),

            match_notes=cls._clean_text(
                payload.get(
                    "match_notes"
                ),
                max_length=1000,
            ),

            appliance_type=cls._clean_text(
                payload.get(
                    "appliance_type"
                ),
                max_length=120,
            ),

            size_value=cls._clean_float(
                payload.get(
                    "size_value"
                )
            ),

            size_unit=cls._clean_text(
                payload.get(
                    "size_unit"
                ),
                max_length=20,
            ),

            color=cls._clean_text(
                payload.get(
                    "color"
                ),
                max_length=80,
            ),

            notes_block=cls._clean_text(
                payload.get(
                    "notes_block"
                ),
                max_length=1800,
            ),

            source_name=cls._clean_text(
                payload.get(
                    "source_name"
                ),
                max_length=200,
            ),

            source_url=cls._clean_text(
                payload.get(
                    "source_url"
                ),
                max_length=2000,
            ),

            confidence=confidence,

            sources=sources[:3],
        )

    @staticmethod
    def _instructions() -> str:

        return """
Research ONE exact appliance model.

Use official manufacturer sources whenever possible.
Never guess.
Never silently substitute another model.

Return structured database fields:
- size_value
- size_unit
- color

IMPORTANT:
Size/capacity and color are stored separately in the ERP.
DO NOT repeat capacity or color in notes_block.

notes_block is a SHORT technical service summary.

STRICT RULE:
Only include fields from the category whitelist below.
Do not add any other specifications even if found.

REFRIGERATOR:
Type
Dimensions = overall Width x Height x Depth only
Weight = net/product weight only
Electrical = voltage / frequency / amps
Ice Maker
Water Dispenser
Water Filter
Refrigerant

AIR CONDITIONER:
Type
Cooling BTU
Coverage
Dimensions = overall Width x Height x Depth only
Weight = net/product weight only
Electrical = voltage / amps
CEER or EER
Refrigerant

WASHER:
Type
Dimensions = overall Width x Height x Depth only
Weight = net/product weight only
Electrical
Load Type
Maximum Spin Speed
Steam

DRYER:
Type
Dimensions = overall Width x Height x Depth only
Weight = net/product weight only
Fuel = Gas or Electric
Electrical
Vent Type
Steam

DISHWASHER:
Type
Dimensions = overall Width x Height x Depth only
Electrical
Sound Level dBA
Place Settings

RANGE / OVEN:
Type
Dimensions = overall Width x Height x Depth only
Fuel = Gas or Electric
Electrical
Oven Capacity
Cooktop Type

MICROWAVE:
Type
Dimensions = overall Width x Height x Depth only
Electrical
Cooking Wattage
Installation Type

FOR ALL CATEGORIES DO NOT INCLUDE:
- capacity in notes if it is returned as size_value
- color or finish
- model confirmation
- model-family explanation
- fresh-food/freezer capacity breakdown
- alternate dimensions
- depth without handle
- depth without door
- door-open dimensions
- shipping weight
- installation clearances
- ENERGY STAR
- annual energy usage
- Wi-Fi
- shelves
- baskets
- drawers
- accessories
- marketing features
- cosmetic descriptions
- installation instructions
- long explanations

Omit a whitelist field if it cannot be verified.
Never replace missing data with guesses.

FORMAT:
Exactly one technical fact per line.

Example refrigerator:

[MODEL SPECS]
Type: French Door / Bottom Freezer
Dimensions: 32.75"W x 69.88"H x 37.5"D
Weight: 253 LB
Electrical: 120V / 60Hz / 15A
Ice Maker: Yes
Water Dispenser: Internal
Water Filter: XWFE
Refrigerant: R600A
[/MODEL SPECS]

Do not add text before or after those technical lines
inside notes_block.

Use maximum 3 useful sources.
match_notes must be one short sentence only.
""".strip()

    @staticmethod
    def _input_text(
        *,
        brand: str,
        model: str,
        serial: str | None,
        appliance_type: str | None,
    ) -> str:

        parts = [
            f"Brand: {brand}",
            f"Model: {model}",
        ]

        if serial:
            parts.append(
                f"Serial: {serial}"
            )

        if appliance_type:
            parts.append(
                "Known appliance category: "
                f"{appliance_type}"
            )

        parts.append(
            """
Research this exact appliance model.

Return verified specifications only.
""".strip()
        )

        return "\n".join(parts)

    def find_specs(
        self,
        *,
        brand: str,
        model: str,
        serial: str | None = None,
        appliance_type: str | None = None,
    ) -> ModelSpecsResult:

        brand = (
            self._clean_text(
                brand,
                max_length=120,
            )
            or ""
        ).upper()

        model = (
            self._clean_text(
                model,
                max_length=120,
            )
            or ""
        ).upper()

        serial = self._clean_text(
            serial,
            max_length=120,
        )

        appliance_type = self._clean_text(
            appliance_type,
            max_length=120,
        )

        if not brand:
            raise ValueError(
                "Brand is required."
            )

        if not model:
            raise ValueError(
                "Model is required."
            )

        schema = {
            "type": "object",
            "properties": {
                "exact_model_confirmed": {
                    "type": "boolean"
                },
                "confirmed_model": {
                    "type": [
                        "string",
                        "null",
                    ]
                },
                "suggested_model": {
                    "type": [
                        "string",
                        "null",
                    ]
                },
                "match_notes": {
                    "type": [
                        "string",
                        "null",
                    ]
                },
                "appliance_type": {
                    "type": [
                        "string",
                        "null",
                    ]
                },
                "size_value": {
                    "type": [
                        "number",
                        "null",
                    ]
                },
                "size_unit": {
                    "type": [
                        "string",
                        "null",
                    ]
                },
                "color": {
                    "type": [
                        "string",
                        "null",
                    ]
                },
                "notes_block": {
                    "type": [
                        "string",
                        "null",
                    ]
                },
                "source_name": {
                    "type": [
                        "string",
                        "null",
                    ]
                },
                "source_url": {
                    "type": [
                        "string",
                        "null",
                    ]
                },
                "confidence": {
                    "type": "string",
                    "enum": [
                        "high",
                        "medium",
                        "low",
                    ],
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string"
                            },
                            "url": {
                                "type": "string"
                            },
                        },
                        "required": [
                            "name",
                            "url",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "exact_model_confirmed",
                "confirmed_model",
                "suggested_model",
                "match_notes",
                "appliance_type",
                "size_value",
                "size_unit",
                "color",
                "notes_block",
                "source_name",
                "source_url",
                "confidence",
                "sources",
            ],
            "additionalProperties": False,
        }

        response = self.client.responses.create(
            model=self.MODEL,

            # Production fast profile.
            # GPT-5.5 defaults to medium reasoning;
            # low is better for this latency-sensitive lookup.
            reasoning={
                "effort": "low",
            },

            # Hard limit on hosted web-search calls.
            max_tool_calls=2,

            # Includes visible + reasoning tokens.
            max_output_tokens=1200,
            tools=[
                {
                    "type": "web_search"
                }
            ],

            instructions=self._instructions(),

            input=self._input_text(
                brand=brand,
                model=model,
                serial=serial,
                appliance_type=appliance_type,
            ),

            text={
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "wccr_model_specs",
                    "strict": True,
                    "schema": schema,
                }
            },
        )

        raw_text = (
            response.output_text
            or ""
        ).strip()

        if not raw_text:
            raise RuntimeError(
                "OpenAI returned an empty response."
            )

        try:
            payload = json.loads(
                raw_text
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenAI returned invalid JSON."
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise RuntimeError(
                "OpenAI response is not a JSON object."
            )

        result = self._normalize_result(
            brand=brand,
            model=model,
            payload=payload,
        )

        if result.notes_block:

            if not result.notes_block.startswith(
                "[MODEL SPECS]"
            ):
                raise RuntimeError(
                    "notes_block is missing "
                    "[MODEL SPECS]."
                )

            if not result.notes_block.endswith(
                "[/MODEL SPECS]"
            ):
                raise RuntimeError(
                    "notes_block is missing "
                    "[/MODEL SPECS]."
                )

        return result
