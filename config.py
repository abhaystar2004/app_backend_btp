import os


class BaseConfig:

	# Paths
	BASE_DIR = os.path.abspath(os.path.dirname(__file__))
	UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
	REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
	MODELS_FOLDER = os.path.join(BASE_DIR, "models")
	IMAGE_METADATA_FILE = os.path.join(UPLOAD_FOLDER, "image_metadata.json")

	# Upload restrictions
	ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}
	MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

	# CORS
	CORS_ALLOW_ALL = True

	# Default modality if caller does not provide one.
	DEFAULT_MODALITY = "oct"
	FUNDUS_CLASS_NAMES = [
		"DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
		"CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
		"RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "PRH",
		"MNF", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA", "VS", "BRAO",
		"PLQ", "HPED", "CL",
	]
	CLASS_FULL_NAMES_BY_MODALITY = {
		"oct": {
			"AMD": "Age-Related Macular Degeneration",
			"CNV": "Choroidal Neovascularization",
			"CSR": "Central Serous Retinopathy",
			"DME": "Diabetic Macular Edema",
			"DR": "Diabetic Retinopathy",
			"DRUSEN": "Drusen",
			"MH": "Macular Hole",
			"NORMAL": "Normal",
		},
		"fundus": {
			"DR": "Diabetic retinopathy",
			"ARMD": "Age-related macular degeneration",
			"MH": "Media haze",
			"DN": "Drusen",
			"MYA": "Myopia",
			"BRVO": "Branch retinal vein occlusion",
			"TSLN": "Tessellation",
			"ERM": "Epiretinal membrane",
			"LS": "Laser scars",
			"MS": "Macular scars",
			"CSR": "Central serous retinopathy",
			"ODC": "Optic disc cupping",
			"CRVO": "Central retinal vein occlusion",
			"TV": "Tortuous vessels",
			"AH": "Asteroid hyalosis",
			"ODP": "Optic disc pallor",
			"ODE": "Optic disc edema",
			"ST": "Optociliary shunt",
			"AION": "Anterior ischemic optic neuropathy",
			"PT": "Parafoveal telangiectasia",
			"RT": "Retinal traction",
			"RS": "Retinitis",
			"CRS": "Chorioretinitis",
			"EDN": "Exudation",
			"RPEC": "Retinal pigment epithelium changes",
			"MHL": "Macular hole",
			"RP": "Retinitis pigmentosa",
			"CWS": "Cotton-wool spots",
			"CB": "Coloboma",
			"PRH": "Preretinal hemorrhage",
			"MNF": "Myelinated nerve fibers",
			"CRAO": "Central retinal artery occlusion",
			"TD": "Tilted disc",
			"CME": "Cystoid macular edema",
			"PTCR": "Post-traumatic choroidal rupture",
			"CF": "Choroidal folds",
			"VH": "Vitreous hemorrhage",
			"MCA": "Macroaneurysm",
			"VS": "Vasculitis",
			"BRAO": "Branch retinal artery occlusion",
			"PLQ": "Plaque",
			"HPED": "Hemorrhagic pigment epithelial detachment",
			"CL": "Collateral",
		},
	}

	# Multi-model specs by modality. Add new modalities here only.
	MODEL_SPECS = {
		"oct": {
			"service": "oct_classifier",
			"framework": "tensorflow",
			"task": "classification",
			"model_path": os.path.join(MODELS_FOLDER, "oct_classifier.h5"),
			"class_names": ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"],
			"input_size": [299, 299],
		},
		"fundus": {
			"service": "fundus_classifier",
			"framework": "pytorch",
			"task": "classification",
			"model_path": os.path.join(MODELS_FOLDER, "convnext46_rfmid.pt"),
			"class_names": FUNDUS_CLASS_NAMES,
			"input_size": [224, 224],
			"multi_label": True,
			"positive_threshold": 0.5,
			"normalize_mean": [0.485, 0.456, 0.406],
			"normalize_std": [0.229, 0.224, 0.225],
		},
	}


Config = BaseConfig


