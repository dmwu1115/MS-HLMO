import cv2

class FeatureMatching():
    def __init__(self):
        self.matcher = cv2.BFMatcher()

    def match(self, desc1s: list, desc2s: list, K_ratio=0.7):
        """
        匹配描述子并滤除异常值
        :param desc1s: image1的高斯金字塔所有层的descriptors
        :param desc2s: image2的高斯金字塔所有层的descriptors
        :return: 1->2 的所有层的matches
        """
        allmatches = [ [] for i in range(len(desc1s))]
        for i, desc1 in enumerate(desc1s):
            for j, desc2 in enumerate(desc2s):
                knnmatches_i_j = self.matcher.knnMatch(desc1, desc2, 2)
                matches_i_j = []
                for pair in knnmatches_i_j:
                    if pair[0].distance < K_ratio * pair[1].distance:
                        matches_i_j.append(pair[0])
                allmatches[i].append(matches_i_j)

        return allmatches

    def remove_outlier_matches(self, allmatches, octave1, layer1, octave2, layer2):
        good_matches = []
        for o1 in range(octave1):
            o1_index = o1 * layer1
            o1_matches = allmatches[o1_index: o1_index + layer1]
            o1_matches_new = self.__union_layer_matches(o1_matches, octave2, layer2)
            good_matches.append(o1_matches_new)

        res = self.__union_layer_matches(good_matches, 1, octave2)[0]
        return res

    # union the layers of each octave
    def __union_layer_matches(self, octave1_matches, octave2_num, layer2_num):
        """
        union the layers of each octave
        :param octave1_matches: pyramid1->pyramid2中的所有匹配
        :param octave2_num: pyramid2中的octave数
        :param layer2_num: pyramid2中每个octave层中的layer数
        :return:
        """

        # 先合并到1->2s中的匹配
        union_1_2s_matches = []
        for i, layer1_2s_matches in enumerate(octave1_matches):
            i_union_matches = []
            for o2 in range(octave2_num):
                o2_index = o2 * layer2_num
                layer1_o2_layers_matches = layer1_2s_matches[o2_index: o2_index + layer2_num]
                temp_union_matches = []
                for layer1_o2_layer_matches in layer1_o2_layers_matches:
                    temp_union_matches += layer1_o2_layer_matches
                temp_union_matches = sorted(temp_union_matches, key=lambda x: x.queryIdx)

                # 选择distance最小的匹配
                union_matches = []
                for m in temp_union_matches:
                    if not union_matches:
                        union_matches.append(m)
                    else:
                        last_m = union_matches[len(union_matches) - 1]
                        if last_m.queryIdx == m.queryIdx:
                            if last_m.distance > m.distance:
                                union_matches[len(union_matches) - 1] = m
                        else:
                            union_matches.append(m)
                i_union_matches.append(union_matches)
            union_1_2s_matches.append(i_union_matches)

        # 合并1s->2中的匹配
        result = []
        for o2 in range(octave2_num):
            temp_union_matches = []
            for i in range(len(union_1_2s_matches)):
                temp_union_matches += union_1_2s_matches[i][o2]
            temp_union_matches = sorted(temp_union_matches, key=lambda x: x.queryIdx)

            union_1s_2_matches = []
            for m in temp_union_matches:
                if not union_1s_2_matches:
                    union_1s_2_matches.append(m)
                else:
                    last_m = union_1s_2_matches[len(union_1s_2_matches) - 1]
                    if last_m.queryIdx == m.queryIdx:
                        if last_m.distance > m.distance:
                            union_1s_2_matches[len(union_1s_2_matches) - 1] = m
                    else:
                        union_1s_2_matches.append(m)
            result.append(union_1s_2_matches)

        return result