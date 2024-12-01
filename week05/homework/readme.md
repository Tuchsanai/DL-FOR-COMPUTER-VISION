# การจดจำใบหน้าของไอดอล K-Pop จาก YouTube

---

การบ้านนี้มุ่งเน้นไปที่การพัฒนาระบบจดจำใบหน้า (Face Recognition) สำหรับไอดอล K-Pop โดยใช้ชุดข้อมูลจาก Kaggle ที่สามารถดาวน์โหลดได้จากลิงก์ต่อไปนี้: [Kaggle Dataset - Kpop Idol Faces](https://www.kaggle.com/datasets/rossellison/kpop-idol-faces)

## โครงสร้างของไฟล์ข้อมูล
ชุดข้อมูลมีการจัดเก็บไฟล์ในรูปแบบโครงสร้างดังนี้:



```
├── kpop-idol-faces/     # โฟลเดอร์หลักที่เก็บชุดข้อมูล
│   └── Kpop_Faces/      # โฟลเดอร์ย่อยที่แบ่งตามกลุ่มไอดอล
│       ├── [ชื่อกลุ่ม]/ # ชื่อโฟลเดอร์เป็นชื่อของกลุ่มไอดอล
│       │   └── [ชื่อไอดอล]/ # ภายในโฟลเดอร์กลุ่มจะมีโฟลเดอร์ย่อยที่เก็บภาพใบหน้าของไอดอลแต่ละคน
```

ยกตัวอย่างเช่น

```
kpop-idol-faces
    Kpop_Faces
        4minute
            hyuna
                137641_VQyg.jpg
                137641_tOUI.jpg
                138012_TgCE.jpg
                138012_oDIw.jpg
                198916_BfYz.jpg
                198916_FQIC.jpg
            choerry
                412164_SrMQ.jpg
                412164_izaU.jpg
                412164_yFWC.jpg
                593670_fpnP.jpg
            chuu
                198916_kmFn.jpg
                198916_sQtv.jpg
                286225_hYuV.jpg
                286225_nvYz.jpg
                412164_Qigl.jpg
                593670_cdrJ.jpg
            jinsoul
                198916_WtiJ.jpg
                286225_jwzM.jpg
                412164_KEkQ.jpg
                412164_lKJz.jpg
                412164_tvwV.jpg
                5290000_obls.jpg
                854700_eLwa.jpg
            yves
                1229881_MDuR.jpg
                1230990_ZbVs.jpg
                854700_OqLV.jpg
        Momoland
            hyebin
                286225_agnY.jpg
            nancy
                138012_VFqK.jpg
                198916_iUgg.jpg
                286225_QeWm.jpg
                317790_fuky.jpg
                412164_WwaJ.jpg
                412164_XINX.jpg
                5290000_OsqR.jpg
                594441_AyDX.jpg
            yeonwoo
                198916_XFOG.jpg
                286225_WvjW.jpg
                412164_EFqZ.jpg
```