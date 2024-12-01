

# การจดจำใบหน้าของไอดอล K-Pop จาก YouTube

-- 
การบ้านนี้มุ่งเน้นการพัฒนาระบบ จดจำใบหน้า (Face Recognition) สำหรับไอดอล K-Pop 

วัตถุประสงค์

	1.	การเก็บข้อมูล:
	•	ดาวน์โหลดชุดข้อมูลใบหน้าของไอดอล K-Pop จาก Kaggle
	•	จัดระเบียบข้อมูลในโฟลเดอร์ของแต่ละกลุ่มและแต่ละไอดอล
	2.	การเตรียมข้อมูล:
	•	สำรวจข้อมูลและทำความสะอาดข้อมูลที่อาจไม่สมบูรณ์
	•	เตรียมข้อมูลสำหรับการเทรนและทดสอบโมเดลจดจำใบหน้า
	3.	การจดจำใบหน้า:
	•	เทรนหรือปรับแต่งโมเดลสำหรับการจดจำใบหน้า (เช่น ใช้ FaceNet)
	•	ทดสอบโมเดลกับวิดีโอจาก YouTube
	4.	การประเมินผล:
	•	ทดสอบโมเดลกับข้อมูลที่ไม่เคยเห็นมาก่อนและประเมินผลลัพธ์
	•	สร้างรายงานสรุปผลการทำงานของโมเดล

โครงสร้างไฟล์

.
├── kpop-idol-faces/     # โฟลเดอร์ที่เก็บชุดข้อมูล
│   └── Kpop_Faces/      # โฟลเดอร์ย่อยแบ่งตามกลุ่มและไอดอล

คำแนะนำในการทำการบ้าน

	1.	ติดตั้งสภาพแวดล้อม:
	•	ติดตั้งไลบรารีที่จำเป็น:

pip install -r requirements.txt


	•	ไลบรารีที่จำเป็น:
	•	facenet-pytorch
	•	opendatasets
	•	pandas

	2.	ดาวน์โหลดชุดข้อมูล:
	•	ใช้แพ็กเกจ opendatasets เพื่อดาวน์โหลดข้อมูลใบหน้าไอดอล K-Pop:

import opendatasets as od
od.download("https://www.kaggle.com/datasets/rossellison/kpop-idol-faces")


	3.	สำรวจชุดข้อมูล:
	•	ตรวจสอบโครงสร้างของชุดข้อมูลที่ดาวน์โหลดมา
	•	จัดระเบียบข้อมูลตามกลุ่มและไอดอล
	4.	การพัฒนาโมเดล:
	•	ใช้โมเดลจดจำใบหน้าที่ผ่านการเทรนมาแล้ว (เช่น FaceNet) หรือเทรนโมเดลของคุณเอง
	•	ทำการดึงใบหน้าและจดจำในวิดีโอ
	5.	การทดสอบวิดีโอจาก YouTube:
	•	ดาวน์โหลดวิดีโอจาก YouTube สำหรับการทดสอบ
	•	ใช้ OpenCV หรือเครื่องมืออื่น ๆ ในการตรวจจับและจดจำใบหน้าในวิดีโอ
	6.	ส่งผลลัพธ์:
	•	บันทึกวิดีโอที่ผ่านการประมวลผลพร้อมติดป้ายกำกับใบหน้าในโฟลเดอร์ results/
	•	รวมถึงการวัดผล (accuracy, precision, recall ฯลฯ) ในรายงาน

ตัวอย่างโค้ด

โหลดข้อมูลและจัดระเบียบ

import os
import pandas as pd

# Path ไปยังชุดข้อมูล
base_path = './kpop-idol-faces/Kpop_Faces'

data = {}
if os.path.exists(base_path):
    groups = [group for group in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, group))]
    for group in groups:
        group_path = os.path.join(base_path, group)
        idols = [idol for idol in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, idol))]
        data[group] = idols
    df = pd.DataFrame.from_dict(data, orient='index').transpose()
    df.columns.name = "Idol Group"
else:
    print(f"The directory {base_path} does not exist!")

df

การเตรียมเฟรมจากวิดีโอ YouTube

import cv2
from facenet_pytorch import MTCNN

# เริ่มต้นใช้งาน MTCNN สำหรับการตรวจจับใบหน้า
mtcnn = MTCNN()

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # ตรวจจับใบหน้า
        boxes, _ = mtcnn.detect(frame)
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            face = frame[y1:y2, x1:x2]
            cv2.imwrite(f"{output_dir}/frame_{frame_idx}.jpg", face)
        frame_idx += 1
    cap.release()

process_video("sample_video.mp4", "./results")

การส่งงาน

	1.	ไฟล์ที่ต้องส่ง:
	•	ไฟล์ homework1.ipynb ที่อัปเดตโค้ดและผลลัพธ์ทั้งหมด
	•	รายงานสรุปการทำงานและผลลัพธ์ของโมเดล
	•	ไฟล์เพิ่มเติมใด ๆ ที่จำเป็นสำหรับการทำซ้ำผลลัพธ์
	2.	วันส่งงาน:
	•	กรุณาส่งงานภายในวันที่ [ใส่กำหนดส่งที่นี่]
	3.	ช่องทางการส่งงาน:
	•	อัปโหลดไฟล์ทั้งหมดไปยัง LMS หรือ GitHub Classroom ของคอร์ส

หมายเหตุ

	•	โปรดอ้างอิงแหล่งข้อมูลชุดข้อมูลหากใช้ในโปรเจกต์
	•	ทดลองใช้โมเดลที่ผ่านการเทรนหลากหลายและบันทึกข้อสังเกต
	•	หากมีข้อสงสัยสามารถติดต่ออาจารย์ได้ตลอดเวลา

ขอให้ทุกคนสนุกกับการเรียนรู้เรื่องการจดจำใบหน้าด้วย Computer Vision! 😊