'''
General Unwrap Method
1. skimage.restoration.unwrap_phase
2. multi-frequency based unwrap
'''

from skimage.restoration import unwrap_phase

def skimage_phase_unwrap(phase_map):
    '''
    Documentation: https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.unwrap_phase
    '''
    unwrapped_phase = unwrap_phase(phase_map)
    return unwrapped_phase