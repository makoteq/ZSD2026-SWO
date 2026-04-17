import numpy as np

def get_x_from_line(line, y):

        if isinstance(line, dict):
            m = line.get('m', 0)
            b = line.get('b', 0)
            return m * y + b

        if hasattr(line, 'get_x'):
            return line.get_x(y)
        if isinstance(line, (list, np.ndarray)) and len(line) == 2:
            return line[0] * y + line[1]
        
        return 0    
