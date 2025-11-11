from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
FEATURE_RANGES_PATH = BASE_DIR / "data" / "feature_ranges.json"


def _load_feature_ranges() -> Dict[str, Dict[str, float]]:
    if FEATURE_RANGES_PATH.exists():
        return json.loads(FEATURE_RANGES_PATH.read_text(encoding="utf-8"))
    return {}


SECTION_DESCRIPTIONS: Dict[str, str] = {
    "Datos generales": (
        "Informacion basica del paciente y lugar de procedencia. "
        "Los campos binarios aceptan 1 para Si y 0 para No."
    ),
    "Ocupacion y entorno": (
        "Seleccione el perfil laboral predominante del paciente en los ultimos meses."
    ),
    "Hospitalizacion y signos vitales": (
        "Datos de ingreso y signos vitales clave registrados al momento de la admision."
    ),
    "Sintomas reportados": (
        "Sintomas observados o relatados durante la consulta inicial."
    ),
    "Laboratorio clinico": (
        "Resultados cuantitativos de laboratorio. Ingrese valores dentro de rangos plausibles."
    ),
}

# Manual overrides for specific features (applied after loading ranges).
FEATURE_OVERRIDES: Dict[str, Dict[str, object]] = {
    "age": {
        "min": 0.0,
        "max": 120.0,
        "strict_bounds": True,
    },
}

# Each entry: (feature_id, label, type, description, extra_options)
RAW_SECTIONS: List[Tuple[str, List[Dict[str, object]]]] = [
    (
        "Datos generales",
        [
            {
                "id": "male",
                "label": "Paciente masculino",
                "type": "binary",
                "description": "1 si el paciente se identifica como hombre, 0 en caso contrario.",
            },
            {
                "id": "female",
                "label": "Paciente femenino",
                "type": "binary",
                "description": "1 si el paciente se identifica como mujer, 0 en caso contrario.",
            },
            {
                "id": "age",
                "label": "Edad",
                "type": "number",
                "step": 0.1,
                "unit": "anios",
                "description": "Edad registrada en anios cumplidos (usar decimales para meses).",
            },
            {
                "id": "urban_origin",
                "label": "Procedencia urbana",
                "type": "binary",
                "description": "1 si reside en zona urbana; 0 para zona rural u otros.",
            },
            {
                "id": "rural_origin",
                "label": "Procedencia rural",
                "type": "binary",
                "description": "1 si reside en zona rural; 0 en otros casos.",
            },
        ],
    ),
    (
        "Ocupacion y entorno",
        [
            {
                "id": "homemaker",
                "label": "Oficios del hogar",
                "type": "binary",
                "description": "1 si se dedica principalmente a labores del hogar.",
            },
            {
                "id": "student",
                "label": "Estudiante",
                "type": "binary",
                "description": "1 si actualmente estudia como actividad principal.",
            },
            {
                "id": "professional",
                "label": "Profesional",
                "type": "binary",
                "description": "1 si ejerce una profesion o empleo especializado.",
            },
            {
                "id": "merchant",
                "label": "Comerciante",
                "type": "binary",
                "description": "1 para actividades de comercio o ventas.",
            },
            {
                "id": "agriculture_livestock",
                "label": "Agricultura o ganaderia",
                "type": "binary",
                "description": "1 si trabaja en el sector agropecuario.",
            },
            {
                "id": "various_jobs",
                "label": "Oficios varios",
                "type": "binary",
                "description": "1 para trabajos informales o de servicios generales.",
            },
            {
                "id": "unemployed",
                "label": "Desempleado",
                "type": "binary",
                "description": "1 si se encuentra sin empleo al momento de la consulta.",
            },
        ],
    ),
    (
        "Hospitalizacion y signos vitales",
        [
            {
                "id": "hospitalization_days",
                "label": "Dias de hospitalizacion",
                "type": "integer",
                "description": "Numero total de dias hospitalizado durante el episodio actual.",
            },
            {
                "id": "body_temperature",
                "label": "Temperatura corporal",
                "type": "number",
                "step": 0.1,
                "unit": "Celsius",
                "description": "Temperatura axilar u oral en grados Celsius.",
            },
            {
                "id": "fever",
                "label": "Fiebre confirmada",
                "type": "binary",
                "description": "Generalmente es 1 (si), de acuerdo con el dataset clinico.",
            },
        ],
    ),
    (
        "Sintomas reportados",
        [
            {"id": "headache", "label": "Cefalea", "type": "binary", "description": "Dolor de cabeza presente."},
            {"id": "dizziness", "label": "Mareos", "type": "binary", "description": "Sensacion de mareo o vertigo."},
            {
                "id": "loss_of_appetite",
                "label": "Perdida de apetito",
                "type": "binary",
                "description": "Dificultad o rechazo para alimentarse.",
            },
            {"id": "weakness", "label": "Debilidad", "type": "binary", "description": "Malestar general o astenia."},
            {"id": "myalgias", "label": "Mialgias", "type": "binary", "description": "Dolores musculares generalizados."},
            {"id": "arthralgias", "label": "Artralgias", "type": "binary", "description": "Dolor en articulaciones."},
            {"id": "eye_pain", "label": "Dolor ocular", "type": "binary", "description": "Dolor retro-orbitario o en ojos."},
            {"id": "hemorrhages", "label": "Sangrado", "type": "binary", "description": "Evidencia de hemorragias."},
            {"id": "vomiting", "label": "Vomito", "type": "binary", "description": "Episodios de vomito recientes."},
            {"id": "abdominal_pain", "label": "Dolor abdominal", "type": "binary", "description": "Dolor en la region abdominal."},
            {"id": "chills", "label": "Escalofrios", "type": "binary", "description": "Presencia de escalofrios o temblor."},
            {"id": "hemoptysis", "label": "Hemoptisis", "type": "binary", "description": "Expecoracion con sangre."},
            {"id": "edema", "label": "Edema", "type": "binary", "description": "Hinchazon o acumulacion de liquidos."},
            {"id": "jaundice", "label": "Ictericia", "type": "binary", "description": "Coloracion amarilla en piel u ojos."},
            {"id": "bruises", "label": "Equimosis", "type": "binary", "description": "Moretones o equimosis visibles."},
            {"id": "petechiae", "label": "Petequias", "type": "binary", "description": "Lesiones puntiformes rojas en piel."},
            {"id": "rash", "label": "Erupcion cutanea", "type": "binary", "description": "Brotes o lesiones en la piel."},
            {"id": "diarrhea", "label": "Diarrea", "type": "binary", "description": "Deposiciones liquidas frecuentes."},
            {
                "id": "respiratory_difficulty",
                "label": "Dificultad respiratoria",
                "type": "binary",
                "description": "Disnea o respiracion trabajosa.",
            },
            {"id": "itching", "label": "Prurito", "type": "binary", "description": "Sensacion de picazon o comezon."},
        ],
    ),
    (
        "Laboratorio clinico",
        [
            {"id": "hematocrit", "label": "Hematocrito", "type": "number", "step": 0.1, "unit": "%"},
            {"id": "hemoglobin", "label": "Hemoglobina", "type": "number", "step": 0.1, "unit": "g/dL"},
            {"id": "red_blood_cells", "label": "Globulos rojos", "type": "integer", "unit": "cel/uL"},
            {"id": "white_blood_cells", "label": "Globulos blancos", "type": "integer", "unit": "cel/uL"},
            {"id": "neutrophils", "label": "Neutrofilos", "type": "number", "step": 0.1, "unit": "%"},
            {"id": "eosinophils", "label": "Eosinofilos", "type": "number", "step": 0.1, "unit": "%"},
            {"id": "basophils", "label": "Basofilos", "type": "number", "step": 0.01, "unit": "%"},
            {"id": "monocytes", "label": "Monocitos", "type": "number", "step": 0.1, "unit": "%"},
            {"id": "lymphocytes", "label": "Linfocitos", "type": "number", "step": 0.1, "unit": "%"},
            {"id": "platelets", "label": "Plaquetas", "type": "integer", "unit": "cel/uL"},
            {"id": "AST (SGOT)", "label": "AST (SGOT)", "type": "integer", "unit": "U/L"},
            {"id": "ALT (SGPT)", "label": "ALT (SGPT)", "type": "integer", "unit": "U/L"},
            {
                "id": "ALP (alkaline_phosphatase)",
                "label": "Fosfatasa alcalina (ALP)",
                "type": "integer",
                "unit": "U/L",
            },
            {"id": "total_bilirubin", "label": "Bilirrubina total", "type": "number", "step": 0.01, "unit": "mg/dL"},
            {"id": "direct_bilirubin", "label": "Bilirrubina directa", "type": "number", "step": 0.01, "unit": "mg/dL"},
            {"id": "indirect_bilirubin", "label": "Bilirrubina indirecta", "type": "number", "step": 0.01, "unit": "mg/dL"},
            {"id": "total_proteins", "label": "Proteinas totales", "type": "number", "step": 0.01, "unit": "g/dL"},
            {"id": "albumin", "label": "Albumina", "type": "number", "step": 0.01, "unit": "g/dL"},
            {"id": "creatinine", "label": "Creatinina", "type": "number", "step": 0.01, "unit": "mg/dL"},
            {"id": "urea", "label": "Urea", "type": "number", "step": 0.1, "unit": "mg/dL"},
        ],
    ),
]


def get_feature_schema(feature_order: List[str]) -> Dict[str, object]:
    feature_ranges = _load_feature_ranges()
    schema_by_id: Dict[str, Dict[str, object]] = {}
    sections: List[Dict[str, object]] = []

    for section_name, fields in RAW_SECTIONS:
        rendered_fields: List[Dict[str, object]] = []
        for field in fields:
            feature_id = field["id"]
            base_meta = {
                "id": feature_id,
                "label": field["label"],
                "type": field.get("type", "number"),
                "category": section_name,
                "description": field.get("description", ""),
                "step": field.get("step"),
                "unit": field.get("unit"),
            }

            range_info = feature_ranges.get(feature_id, {})
            if range_info:
                base_meta["min"] = range_info.get("min")
                base_meta["max"] = range_info.get("max")
                base_meta["mean"] = range_info.get("mean")

            override = FEATURE_OVERRIDES.get(feature_id)
            if override:
                base_meta.update(override)

            if base_meta["type"] == "binary":
                base_meta["allowed_values"] = [0, 1]
                base_meta.setdefault("step", 1)
            elif base_meta["type"] == "integer":
                base_meta.setdefault("step", 1)
            else:
                base_meta.setdefault("step", 0.01)

            schema_by_id[feature_id] = base_meta
            rendered_fields.append(base_meta)

        sections.append(
            {
                "name": section_name,
                "description": SECTION_DESCRIPTIONS.get(section_name, ""),
                "items": rendered_fields,
            }
        )

    ordered_features = [fid for fid in feature_order if fid in schema_by_id]
    missing = [fid for fid in feature_order if fid not in schema_by_id]

    for missing_id in missing:
        range_info = feature_ranges.get(missing_id, {})
        fallback = {
            "id": missing_id,
            "label": missing_id.replace("_", " ").title(),
            "type": "number",
            "category": "Otras caracteristicas",
            "description": "",
            "step": 0.01,
            "min": range_info.get("min"),
            "max": range_info.get("max"),
            "mean": range_info.get("mean"),
        }
        override = FEATURE_OVERRIDES.get(missing_id)
        if override:
            fallback.update(override)
        schema_by_id[missing_id] = fallback
        ordered_features.append(missing_id)

    return {
        "order": ordered_features,
        "by_id": schema_by_id,
        "sections": sections,
    }
