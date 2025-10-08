

def get_iris_detector(detector):
    if detector == 'ert':
        from .ert_detector import ert_detector
        return ert_detector
    elif detector == 'peak':
        from .peak_detector import peak_detector
        return peak_detector
    elif detector == 'enet':
        from .enet_detector import enet_detector
        return enet_detector
    else:
        # Try to read custom detector from file.
        try:
            # open the file
            fp = open(detector, 'r')
        except RuntimeError:
            print('Cannot open {}.'.format(detector))
            return None
        try:
            # run the file.
            # custom_detector must be defined in the code.
            code = fp.read()
            fp.close()
            exec(code)
        except RuntimeError:
            print('Error in {}.'.format(detector))
            return None
        try:
            # If custom_detector is defiend, return it.
            return custom_detector
        except RuntimeError:
            print('"custom_detector" is not defined in {}.'.format(detector))
            return None
