def normalize_coord(kpts, width, height):
    kpts = kpts[:]
    for i in range(len(kpts)):
        kpts[i] = kpts[i][0] / width, kpts[i][1] / height
    return kpts

def denormalize_coord(kpts, width, height):
    kpts = kpts[:]
    for i in range(len(kpts)):
        kpts[i] = (int(kpts[i][0] * width), int(kpts[i][1] * height))
    return kpts