"""Class labels and descriptions for the clinical predictor."""

CLASS_DETAILS = {
    1: {
        "label": "Dengue",
        "description": "Infeccion viral transmitida por mosquitos Aedes, suele presentar fiebre alta, dolor muscular y malestar general."
    },
    2: {
        "label": "Malaria",
        "description": "Enfermedad parasitaria transmitida por mosquitos Anopheles, cursa con fiebre en picos, escalofrios y anemia si no se trata a tiempo."
    },
    3: {
        "label": "Leptospirosis",
        "description": "Infeccion bacteriana por contacto con agua o suelos contaminados, puede causar fiebre, dolor abdominal, ictericia y compromiso renal."
    },
}


def get_class_detail(class_id: int) -> dict:
    return CLASS_DETAILS.get(
        int(class_id),
        {
            "label": f"Clase {class_id}",
            "description": "Descripcion no disponible para esta clase.",
        },
    )
