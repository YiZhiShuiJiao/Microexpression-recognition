import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from tkinter.filedialog import askopenfilename
import time
from skimage import io,transform
import tensorflow as tf
import numpy as np

root = Tk()
root.geometry('500x600')
root.title('花卉识别')

p_text = Text(root, width=60, height=30)
p_text.pack(padx=10, pady=10)
p_text.config(state=DISABLED)

def choosepic():
    global path_ 
    path_= askopenfilename() 
    path.set(path_)
    img_open = Image.open(file_entry.get())
    img = ImageTk.PhotoImage(img_open)
    #image_label.config(image=img)
    image_label.image = img  # keep a reference
    p_text.config(state=NORMAL)
    p_text.delete('1.0','end')
    p_text.image_create(END, image=img)
    p_text.config(state=DISABLED)

path = StringVar()  #跟踪变量变化
Button(root, text='选择图片', command=choosepic, font=('Arial', 12), width=10, height=1 ).pack()
file_entry = Entry(root, state='readonly', text=path)
image_label = Label(root)
image_label.pack()

a_text = Text(root, width=60, height=4)
a_text.pack(padx=10, pady=10)
a_text.config(state=DISABLED)

def evaluate_one_image(path):
    path=path

    # 模型目录
    CHECKPOINT_DIR = 'C:/Users/asus/runs/1557125558/checkpoints'  #训练后生成的检查点文件夹，在当前工程下。
    INCEPTION_MODEL_FILE = 'C:/Users/acer/Desktop/a/inception/tensorflow_inception_graph.pb'

    # inception-v3模型参数
    BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
    JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

    flower_dict={0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}

    # 读取数据
    image_data = tf.gfile.FastGFile(path, 'rb').read()

    # 评估
    checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    with tf.Graph().as_default() as graph:
        with tf.Session().as_default() as sess:

            # 读取训练好的inception-v3模型
            with tf.gfile.FastGFile(INCEPTION_MODEL_FILE, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def,
                return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

            # 使用inception-v3处理图片获取特征向量
            bottleneck_values = sess.run(bottleneck_tensor,{jpeg_data_tensor: image_data})
            # 将四维数组压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
            bottleneck_values = [np.squeeze(bottleneck_values)]

            # 加载图和变量（这里我选择的是step=900的图，使用的是绝对路径。）
            saver = tf.train.import_meta_graph('C:/Users/asus/runs/1557125558/checkpoints/model-3700.meta')
            saver.restore(sess, tf.train.latest_checkpoint('C:/Users/asus/runs/1557125558/checkpoints/'))

#            x = graph.get_tensor_by_name("BottleneckInputPlaceholder")
#            feed_dict = {x: bottleneck_values}
#            logits = graph.get_tensor_by_name("evaluation/ArgMax")
#             classification_result = sess.run(logits, feed_dict)
#             # 打印出预测矩阵
#             print(classification_result)
#             # 打印出预测矩阵每一行最大值的索引
#            print(tf.argmax(classification_result, 1).eval())
#            # 根据索引通过字典对应花的分类
#             output = []
#             output = tf.argmax(classification_result, 1).eval()
#            result=print( "朵花预测:" + flower_dict[output])

            # 通过名字从图中获取输入占位符
            input_x = graph.get_operation_by_name('BottleneckInputPlaceholder').outputs[0]

            # 我们想要评估的tensors
            predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]


            # 收集预测值
            all_predictions = []
            all_predictions = sess.run(predictions, {input_x: bottleneck_values})
            #print(all_predictions)

            index = str(all_predictions)[1]

            max_index = int(index)
            if max_index == 0:
                result = ('雏菊' )
            elif max_index == 1:
                result = ('蒲公英' )
            elif max_index == 2:
                result = ('玫瑰' )
            elif max_index == 3:
                result = ('向日葵' )
            else:
                result = ('郁金香' )
          
    return result
            
def dis():
    a_text.config(state=NORMAL)
    a_text.delete('1.0','end')
    a_text.insert(INSERT,"花朵预测："+evaluate_one_image(path_))
    a_text.config(state=DISABLED)
                
Button(root, text='识别', command=dis, font=('Arial', 12), width=10, height=1 ).pack()
root.mainloop()
