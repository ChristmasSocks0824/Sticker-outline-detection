import numpy as np
#使用向量化運算，優勢比迴圈快很多

def array_sum(values):#將傳入的資料，使用Numpy陣列加速計算做sum總和
    arr = np.array(values)
    return int(arr.sum())#快速計算陣列中所有數字的總和

def normalize_byte_values(data):#顏色亮度分配
    arr = np.frombuffer(bytes(data), dtype=np.uint8)#將陣列的數字轉換成32位元(才能做計算)
    arr = arr.astype(np.float32) / 255.0#把arr產生的32位元浮點數/255變成0或1的數字
    return arr.tolist()#最後把0跟1的數字變成列表型態list
    #把運算數字給縮小，0-> 0.0  127->0.498  256->1.0

def make_gradient(width, height):#有顏色後，生成2D漸層圖形
    x = np.linspace(0, 255, width, dtype=np.uint8)#0~255之間平均生成width個數字(橫軸)
    img = np.tile(x, (height, 1))#在垂直方向也生成height個數字
    #形成2d影像圖
    return img.tobytes()