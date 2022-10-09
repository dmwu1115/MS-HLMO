from HS_HLMO.ggloh_describer import *
from HS_HLMO.pmom_generator import *

class KeypointDescriber():
    def __init__(self, P_sigma, NA, NO, R0=5):
        self.feature_extractor = PMOM(P_sigma)
        self.describer = GGLOH(NA, NO, R0)

    def generate_descriptors(self, image, kpts):
        pmom_image = self.feature_extractor.generate_PMOM(image)
        descriptors = self.describer.generate_descriptors(pmom_image, kpts)
        return descriptors