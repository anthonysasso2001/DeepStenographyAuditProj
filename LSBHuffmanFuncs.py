from LSBSteg.LSBSteg import LSBSteg
from dahuffman import HuffmanCodec
from datasets import load_dataset
import matplotlib.pyplot as plt
import os, certifi
import numpy as np
import cv2
import subprocess

# code example from https://github.com/RobinDavid/LSB-Steganography libraru, lsb is using
# LSB 
def LSBEmbed(text,image):
    # Embed text into image using least sig bit method
    steg = LSBSteg(image)

    # print (text)
    img_encoded = steg.encode_binary(text)
    # cv2.imwrite(name, img_encoded)
    return img_encoded

def LSBExtract(image):
    # pull least sig bit from image and reformat into embedded text
    steg = LSBSteg(image)
    extracted_text = steg.decode_binary()
    return extracted_text

# huffman serialization assisted with chatgpt for debugging, dahuffman use from PyPl example
def HuffmanEncode(data):
    # encode text into huffman tree + map
    codec = HuffmanCodec.from_data(data)
    encoded_data = codec.encode(data)

    # finally prepend all so tree:data is one block
    return codec, encoded_data

def HuffmanDecode(codec,data):
    # using Huffman map reconstruct text to original values and decode / unzip data
    decoded_data = codec.decode(data)
    return decoded_data


if __name__ == "__main__":
    os.environ['SSL_CERT_FILE'] = certifi.where()

    #example usage encode/embed
    # for creating stegimage:

    # option 1 uses "Hello World!" for all to reuse codec and make things simpler
    tiny_imageNet = load_dataset("zh-plus/tiny-imagenet")
    ex_data = "Hello World!"
    codec,encoded_data = HuffmanEncode(ex_data)
    path = fr"{os.getcwd()}\data"
    os.makedirs(path,exist_ok=True)
    path+="\secret"
    os.makedirs(path,exist_ok=True)
    
# count len(tiny_imageNet['train'])
    for i in range(0,10):
        image = tiny_imageNet['train'][i]['image']
        np_image = np.array(image)
        embed_img = LSBEmbed(encoded_data,np_image)
        
        print(path)
        cv2.imwrite(fr"{path}\{i}.jpeg",embed_img)
        plt.imshow(embed_img)

    for i in range(0,10):
        image = cv2.imread(path)
        plt.imshow(image)
        # subprocess.run([
        #     "python",
        #     "PyTorch-Deep-Image-Steganography/train.py",
        #     "--cover_dir", "data/cover",
        #     "--secret_dir", "data/secret",
        #     "--epochs", "10",
        #     "--batch_size", "4",
        #     "--lr", "0.0001",
        #     "--save_dir", "checkpoints/"
        # ])

        
        # secret_data = LSBExtract(embed_img)
        # recovered_text = HuffmanDecode(codec,secret_data)

