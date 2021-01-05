from collections import deque
from lenstronomy.Util.numba_util import jit

@jit
def _get_fraction(from_value, to_value, level):
    if (to_value == from_value):
        return 0
    return ((level - from_value) / (to_value - from_value))

@jit
def marching_squares(cells, values):
    # Inspired by scikit-image implementation.
    # Cells: N*4*2
    # Values: N*4
    segments=[]
    for square, m in zip(cells, values):
        level=0

        ul = m[1]
        ur = m[3]
        ll = m[0]
        lr = m[2]
        r0, c0 = square[0]
        r1, c1 = square[3]

        # Skip this square if any of the four input values are NaN.
        if np.any(np.isnan(square)):
            continue

        square_case = 0
        if (ul > level): square_case += 1
        if (ur > level): square_case += 2
        if (ll > level): square_case += 4
        if (lr > level): square_case += 8

        if square_case in [0, 15]:
            # only do anything if there's a line passing through the
            # square. Cases 0 and 15 are entirely below/above the contour.
            continue

        top = r0 + _get_fraction(ul, ur, level)*(r1-r0), c1
        bottom = r0 + _get_fraction(ll, lr, level)*(r1-r0), c0
        left = r0, c0 + _get_fraction(ll, ul, level)*(c1-c0)
        right = r1, c0 + _get_fraction(lr, ur, level)*(c1-c0)

        if (square_case == 1):
            # top to left
            segments.append((top, left))
        elif (square_case == 2):
            # right to top
            segments.append((right, top))
        elif (square_case == 3):
            # right to left
            segments.append((right, left))
        elif (square_case == 4):
            # left to bottom
            segments.append((left, bottom))
        elif (square_case == 5):
            # top to bottom
            segments.append((top, bottom))
        elif (square_case == 6):
            raise ValueError("Bad marching squares topology - something wrong with grid.")
        elif (square_case == 7):
            # right to bottom
            segments.append((right, bottom))
        elif (square_case == 8):
            # bottom to right
            segments.append((bottom, right))
        elif (square_case == 9):
            raise ValueError("Bad marching squares topology - something wrong with grid.")
        elif (square_case == 10):
            # bottom to top
            segments.append((bottom, top))
        elif (square_case == 11):
            # bottom to left
            segments.append((bottom, left))
        elif (square_case == 12):
            # lef to right
            segments.append((left, right))
        elif (square_case == 13):
            # top to right
            segments.append((top, right))
        elif (square_case == 14):
            # left to top
            segments.append((left, top))
    return segments

def _assemble_contours(segments):
    # Almost entirely from the scikit-image implementation.
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    for from_point, to_point in segments:
        # Ignore degenerate segments.
        # This happens when (and only when) one vertex of the square is
        # exactly the contour level, and the rest are above or below.
        # This degenerate vertex will be picked up later by neighboring
        # squares.
        if from_point == to_point:
            continue

        tail, tail_num = starts.pop(to_point, (None, None))
        head, head_num = ends.pop(from_point, (None, None))

        if tail is not None and head is not None:
            # We need to connect these two contours.
            if tail == head:
                # We need to closed a contour.
                # Add the end point
                head.append(to_point)
            else:  # tail is not head
                # We need to join two distinct contours.
                # We want to keep the first contour segment created, so that
                # the final contours are ordered left->right, top->bottom.
                if tail_num > head_num:
                    # tail was created second. Append tail to head.
                    head.extend(tail)
                    # remove all traces of tail:
                    ends.pop(tail[-1])
                    contours.pop(tail_num, None)
                    # Update contour starts end ends
                    starts[head[0]] = (head, head_num)
                    ends[head[-1]] = (head, head_num)
                else:  # tail_num <= head_num
                    # head was created second. Prepend head to tail.
                    tail.extendleft(reversed(head))
                    # remove all traces of head:
                    starts.pop(head[0])
                    contours.pop(head_num, None)
                    # Update contour starts end ends
                    starts[tail[0]] = (tail, tail_num)
                    ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            # we need to add a new contour
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:  # tail is not None
            # We've found a single contour to which the new segment should be
            # prepended.
            tail.appendleft(from_point)
            starts[from_point] = (tail, tail_num)
        else:  # tail is None and head is not None:
            # We've found a single contour to which the new segment should be
            # appended
            head.append(to_point)
            ends[to_point] = (head, head_num)

    return [np.array(contour) for _, contour in sorted(contours.items())]
