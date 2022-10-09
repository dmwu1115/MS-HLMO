import cv2

class FeatureMatching():
    def __init__(self):
        self.matcher = cv2.BFMatcher()

    def match(self, desc1s: list, desc2s: list):
        """
        匹配描述子并滤除异常值
        :param desc1s: image1的高斯金字塔所有层的descriptors
        :param desc2s: image2的高斯金字塔所有层的descriptors
        :return: 1->2 的所有层的matches
        """
        allmatches = [ [] for i in range(len(desc1s))]
        for i, desc1 in enumerate(desc1s):
            for j, desc2 in enumerate(desc2s):
                matches_i_j = self.matcher.match(desc1, desc2)
                allmatches[i].append(matches_i_j)

        return allmatches

    # TODO: Remove outlier matches
    def remove_outlier_matches(self, allmatches, octave1, layer1, octave2, layer2):
        pass
