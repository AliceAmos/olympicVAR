FRAMES_PER_VID = 103

def calc_somersaults(landmarks):                                                                                        # calculate amount of somersalts accordint to landmarks' list

    counter = 0
    inSault = False
    for LMlist in landmarks:                                                                                            # go over the landmarks
        if len(LMlist.list) > 0:
            if LMlist.list[1][1] > LMlist.list[11][1]:                                                                  # if the head is above the legs
                if inSault:                                                                                             # and a sommersault has already started, this is a full round
                    counter += 1
                inSault = False
            else:
                inSault = True                                                                                          # if the head is below them, a sommersault has begun

    if inSault:                                                                                                         # in case ended without finishing a full round
        counter += 0.5

    return counter
