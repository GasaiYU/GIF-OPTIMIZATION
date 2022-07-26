def two_D_multiply(img1: list, img2: list):
    """
    img1 (list): _description_
    img2 (list): _description_
    return: The multiplition of every elem in img1 and img2.
    """
    result_img = []
    row = []
    for i in range(len(img1)):
        row = []
        for j in range(len(img1[i])):
            row.append(img1[i][j] * img2[i][j])
        result_img.append(row)

    return result_img

def two_D_minus(img1: list, img2: list):
    """
    Args:
        img1 (list): _description_
        img2 (list): _description_
        return: Every pixel in img1 minus every pixel in img2
    """ 
    result_img = []
    row = []
    for i in range(len(img1)):
        row = []
        for j in range(len(img1[i])):
            row.append(img1[i][j] - img2[i][j])
        result_img.append(row)

    return result_img

def two_D_add(img1: list, img2: list):
    """
    Args:
        img1 (list): _description_
        img2 (list): _description_
        return: Every pixel in img1 add every pixel in img2
    """ 
    result_img = []
    row = []
    for i in range(len(img1)):
        row = []
        for j in range(len(img1[i])):
            row.append(img1[i][j] + img2[i][j])
        result_img.append(row)

    return result_img

def two_D_divide(img1: list, img2: list):
    """
    Args:
        img1 (list): _description_
        img2 (list): _description_
        return: Every pixel in img1 divide every pixel in img2
    """ 
    result_img = []
    row = []
    for i in range(len(img1)):
        row = []
        for j in range(len(img1[i])):
            row.append(img1[i][j] / img2[i][j])
        result_img.append(row)

    return result_img

def blur(img: list, radius: int):
    """
    img (list): The image which needs filtering
    radius (int): The filtering box window's radius.
    return: The filtered image.
    """
    assert radius < len(img) and radius < len(img[0])

    img_clone = img.copy()
    padding_img = padding(img_clone, radius)

    blur_img = []
    blur_row = []
    for i in range(radius, radius + len(img)):
        blur_row = []
        for j in range(radius, radius + len(img[0])):
            blur_row.append(mean_box(padding_img, i, j, radius))
        blur_img.append(blur_row)
    
    return blur_img

def seperable_blur(img: list, radius: int):
    """
    img (list): The image which needs filtering
    radius (int): The filtering box window's radius.
    return: The filtered image.
    """
    assert radius < len(img) and radius < len(img[0])
    
    img_clone = img.copy()
    padding_img = padding(img_clone, radius)
    
    blur_img = []
    
    # The first conv.
    for i in range(radius, len(img) + radius):
        blur_row = []
        for j in range(radius, radius + len(img[0])):
            blur_row.append(seperable_mean_box(padding_img, i, j, radius, 1))
        blur_img.append(blur_row)


    # The second conv
    img_clone = padding(blur_img, radius)

    blur_img = []
    for i in range(radius, radius + len(img)):
        blur_row = []
        for j in range(radius, radius + len(img[0])):
            blur_row.append(seperable_mean_box(img_clone, i, j, 1, radius))
        blur_img.append(blur_row)

    return blur_img

def padding(img: list, radius:int):
    """
    img (list): The image needed padding
    radius (int): The padding radius.
    return: The image after padding.
    """
    padding_img = []
    padding_row = []
    for i in range(-radius, len(img) + radius):
        padding_row = []
        for j in range(-radius, len(img[0]) + radius):
            if i < 0 or i >= len(img) or j < 0 or j >= len(img):
                padding_row.append(0)
            else:
                padding_row.append(img[i][j])
        padding_img.append(padding_row)
        
    return padding_img

def mean_box(img: list, x_pos: int, y_pos: int, radius: int):
    """
    Args:
        img (list): The image needed dealing with.
        x_pos (int): The left most pixel's x_pos.
        y_pos (int): The left most pixel's y_pos.
        radius (int): The radius of the box.
    Return:
        The mean of the box's value.
    """
    assert radius > 0
    
    sum_value = 0
    for i in range(radius):
        for j in range(radius):
            sum_value += img[x_pos+i][y_pos+j]
    
    return sum_value / (radius**2)
  
def seperable_mean_box(img: list, x_pos: int, y_pos: int, h: int, w: int):
    """
    Args:
        img (list): The image needed dealing with.
        x_pos (int): The left most pixel's x_pos.
        y_pos (int): The left most pixel's y_pos.
        h (int): The height of the box.
        w (int): The width of the box.
    Return:
        The mean of the box's value.
    """
    assert h > 0 and w > 0
    
    sum_value = 0
    for i in range(h):
        for j in range(w):
            sum_value += img[x_pos+i][y_pos+j]
            
    return sum_value / (h * w)

class GuidedFilter:

    def __init__(self, I, radius=3, epsilon=0.4):
        """
        :param I: input image
        :param radius: The radius of the window
        :param epsilon: Value controlling sharpness
        """
        self.filter = GrayGuidedFilter(I, radius, epsilon)
        
        
        
    def filt(self, p):
        """
        :param p: Guided image
        :return: Filtering image
        """
        return self.filter.filter(p)


blur_func = seperable_blur

class GrayGuidedFilter:
   
    def __init__(self, I, radius, epsilon):
        """
        :param I: input image
        :param radius: The radius of the window
        :param epsilon: Value controlling sharpness
        """
        self.I = I
        self.radius = radius
        self.epsilon = epsilon

    def filter(self, p):
        """
        :param p: The guided image.
        :return: Filtering output.
        """
        I = self.I
        r = self.radius
        eps = self.epsilon
        
        # step 1
        mean_I = blur_func(I, r)
        mean_p = blur_func(p, r)
        corr_I = blur_func(two_D_multiply(I, I), r)
        corr_IP = blur_func(two_D_multiply(I, p), r)

        # step 2
        var_I = two_D_minus(corr_I, two_D_multiply(mean_I, mean_I))
        cov_Ip = two_D_minus(corr_IP, two_D_multiply(mean_I, mean_p))

        # step 3
        fill_row = []
        fill_img = []
        for i in range(len(var_I)):
            fill_row = []
            for j in range(len(var_I[0])):
                fill_row.append(eps)
            fill_img.append(fill_row)
        
        a = two_D_divide(cov_Ip, two_D_add(var_I, fill_img))
        b = two_D_minus(mean_p, two_D_multiply(a, mean_I))

        # step 4
        mean_a = blur_func(a, r)
        mean_b = blur_func(b, r)

        # step 5
        q = two_D_add(two_D_multiply(mean_a, I), mean_b)
        return q
    


class ColorGuidedFilter:
    def __init__(self, I, radius, epsilon):
        """
        :param I: input image
        :param radius: The radius of the window
        :param epsilon: Value controlling sharpness
        """
        pass

    def filter(self, p):
        pass


