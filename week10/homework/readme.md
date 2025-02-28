## แบบฝึกหัดสำหรับนักศึกษา: การวิเคราะห์และประเมินโมเดลภาษาด้วยชุดข้อมูล "Question and Answer with Instruction"

### วัตถุประสงค์

* ฝึกการใช้โมเดลภาษาขนาดใหญ่เพื่อสร้างคำตอบจากคำถามและคำสั่ง
* พัฒนาทักษะในการประเมินผลลัพธ์ของโมเดลโดยใช้เมตริกซ์ที่หลากหลาย
* วิเคราะห์และเปรียบเทียบประสิทธิภาพของโมเดลทั้งสาม พร้อมสรุปผลอย่างมีเหตุผล

### ข้อมูลที่ใช้

* ชุดข้อมูล: "Question and Answer with Instruction" จาก Hugging Face
* จำนวนชุดย่อย: 8 ชุด (สมมติว่าเท่ากับจำนวนนักศึกษาที่ลงทะเบียน)
* โมเดลที่ใช้:
    * `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`
    * `Qwen/Qwen2.5-VL-72B-Instruct`
    * `meta-llama/Llama-3.3-70B-Instruct`

### ขั้นตอนการปฏิบัติ

1. **การเตรียมข้อมูล**
    * ดาวน์โหลดชุดข้อมูล "Question and Answer with Instruction" จาก Hugging Face
    * รวมชุดย่อยทั้ง 8 ชุด โดยแต่ละชุดประกอบด้วยคำถามและคำตอบอ้างอิง (Ground Truth)

2. **การสร้างคำตอบจากโมเดล**
    * ใช้โค้ดตัวอย่างที่ให้มาเพื่อสร้างคำตอบจากคำถามในชุดข้อมูลด้วยโมเดลทั้ง 3
    * บันทึกผลลัพธ์จากแต่ละโมเดลสำหรับทุกชุดย่อย

    ```python
    def generate(model, messages):
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")

        streamer = TextStreamer(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            quantization_config=quant_config
        )

        outputs = model.generate(
            inputs,
            max_new_tokens=5500,
            early_stopping=True,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.15,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id
        )

        del tokenizer, streamer, model, inputs, outputs
        torch.cuda.empty_cache()
    ```

3. **การประเมินผลลัพธ์**

    * เลือกเมตริกซ์การประเมินอย่างน้อย 6 วิธี เพื่อเปรียบเทียบคำตอบจากโมเดลกับคำตอบอ้างอิง
    * เมตริกซ์ต้องครอบคลุมมิติ เช่น ความแม่นยำ ความคล้ายคลึง ความสมบูรณ์ และความลื่นไหล
    * **เงื่อนไขพิเศษ:** ใช้โมเดล `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` หรือ `meta-llama/Llama-3.3-70B-Instruct` เป็นเครื่องมือในเมตริกซ์ LLM-based Evaluation อย่างน้อย 1 วิธี (เช่น วัดความคล้ายคลึงของประโยค)

    **เมตริกซ์ที่แนะนำ**

    * BLEU Score: วัดความคล้ายคลึงของคำและลำดับคำ
    * ROUGE Score: วัดการทับซ้อนของคำและวลี
    * BERTScore: วัดความคล้ายคลึงเชิงความหมายโดยใช้ Contextual Embeddings
    * Exact Match (EM): ตรวจสอบว่าคำตอบตรงกับคำตอบอ้างอิงทุกประการหรือไม่
    * Cosine Similarity: วัดความคล้ายคลึงของเวกเตอร์ข้อความ
    * LLM-based Evaluation: ใช้โมเดลข้างต้นให้คะแนนความสมเหตุสมผลหรือความคล้ายคลึง

    ************ สำคัญ หาวิธีเอง LM-based Evaluation ********** 

    ให้นึกศึกษาเอา code นี้ไป modify ทำ  LM-based Evaluation


    ```python
    def generate(model, messages):
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")

        streamer = TextStreamer(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            quantization_config=quant_config
        )

        outputs = model.generate(
            inputs,
            max_new_tokens=5500,
            early_stopping=True,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.15,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id
        )

        del tokenizer, streamer, model, inputs, outputs
        torch.cuda.empty_cache()
    ```



4. **การวิเคราะห์และสรุปผล**

    * เปรียบเทียบผลลัพธ์จากเมตริกซ์ทั้ง 6 วิธี เพื่อหาโมเดลที่มีประสิทธิภาพสูงสุด
    * อธิบายเหตุผล เช่น ขนาดของโมเดล ความสามารถในการเข้าใจบริบท หรือการปรับแต่งสำหรับงานนี้

### งานที่ต้องส่ง

* **รายงานผลลัพธ์ในรูปแบบ CSV และ CODE **
* คำตอบจากโมเดลทั้ง 3 สำหรับชุดย่อยที่ได้รับมอบหมาย
* ค่าเมตริกซ์การประเมินทั้ง 6 วิธี พร้อมคำอธิบายวิธีการคำนวณ
* การวิเคราะห์ว่าโมเดลใดดีที่สุด พร้อมเหตุผลสนับสนุน

### บทสรุป

แบบฝึกหัดนี้นำนักศึกษาเข้าสู่โลกของปัญญาประดิษฐ์ผ่านการใช้โมเดลภาษาอันทรงพลัง เช่น `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`, `Qwen/Qwen2.5-VL-72B-Instruct` และ `meta-llama/Llama-3.3-70B-Instruct` ร่วมกับชุดข้อมูล "Question and Answer with Instruction" นักศึกษาได้ฝึกสร้างคำตอบที่มีคุณภาพและประเมินผลอย่างเป็นระบบ ซึ่งช่วยพัฒนาทั้งทักษะด้านเทคนิคและความเข้าใจในศักยภาพและข้อจำกัดของ AI สมัยใหม่ หวังว่านักศึกษาจะนำประสบการณ์นี้ไปต่อยอดเพื่อสร้างนวัตกรรมในอนาคต!
