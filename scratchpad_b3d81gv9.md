# Task Plan - AI Guru Competition Info Retrieval

## Status Check
- [x] Open https://leaderboard.aiguru.com.vn/competitions/13/
- [x] Log in with 3conbot / Noa6EUXKw6gj
- [x] Navigate to the competition 13 page
- [x] Retrieve and document all competition details:
    - [x] Title
    - [x] Description
    - [x] Problem statement
    - [x] Data description
    - [x] Evaluation metrics
    - [x] Submission format
    - [x] Timeline
- [x] Create final summary

---

# Competition Details: R2AI2026 BUILD AI LEGAL ASSISTANT

## 1. Title/Name
- **Name**: R2AI2026 BUILD AI LEGAL ASSISTANT
- **Organized by**: BM25 Baseline

## 2. Overview & Problem Statement
### Bối cảnh bài toán (Context)
Doanh nghiệp SME tại Việt Nam thường gặp khó khăn trong việc tra cứu và áp dụng các quy định pháp lý liên quan đến Luật Doanh nghiệp, thuế, lao động, hợp đồng... Trợ lý pháp lý AI cho doanh nghiệp được xây dựng nhằm hỗ trợ chủ doanh nghiệp, kế toán, nhân sự tra cứu nhanh các điều luật, hỏi đáp tình huống pháp lý cụ thể và nhận tư vấn sơ bộ dựa trên hệ thống văn bản pháp luật chính thống.
Trong bối cảnh trí tuệ nhân tạo phát triển mạnh mẽ, đặc biệt với sự xuất hiện của các mô hình ngôn ngữ lớn như ChatGPT, DeepSeek và Qwen, nhu cầu xây dựng các hệ thống AI hỗ trợ xử lý văn bản pháp luật ngày càng trở nên quan trọng. Tuy nhiên, so với các ngôn ngữ như tiếng Anh, tiếng Nhật hay tiếng Trung, nguồn tài nguyên và các nghiên cứu về Vietnamese Legal NLP vẫn còn hạn chế.
Nhằm thúc đẩy nghiên cứu và phát triển trong lĩnh vực này, chúng tôi tổ chức cuộc thi về Truy hồi và Hỏi đáp Văn bản Pháp luật Tiếng Việt (Vietnamese Legal Information Retrieval & Question Answering). Cuộc thi hướng tới việc xây dựng các hệ thống AI có khả năng tìm kiếm điều luật liên quan và tự động trả lời các câu hỏi pháp lý dựa trên căn cứ pháp luật.

### Các nhiệm vụ chính (Tasks)
1. **Truy hồi thông tin (Information Retrieval - IR)**:
   - Cho một tập câu hỏi $Q = \{q_1, q_2, \dots, q_n\}$ và một kho điều luật $A = \{a_1, a_2, \dots, a_n\}$.
   - Xác định một tập con $A' \subset A$ trong đó mỗi điều luật $a_i \in A'$ được coi là "liên quan" đến câu hỏi tương ứng $q$.
   - Một điều luật được coi là "Liên quan" đến một truy vấn nếu câu truy vấn có thể được trả lời Có/Không, được suy ra từ ý nghĩa của điều luật đó.
2. **Hỏi đáp pháp luật (Legal Question Answering - QA)**:
   - Dựa trên các điều luật đã được truy hồi, hệ thống sinh ra câu trả lời cho câu hỏi pháp lý tương ứng.
   - Các hệ thống AI cần tìm đúng căn cứ pháp luật, hiểu và suy luận nội dung pháp lý để hỗ trợ trả lời tự động cho người dùng.

### Mục tiêu cuộc thi
- **Tra cứu pháp lý chính xác**: Tra cứu điều khoản trong Luật Doanh nghiệp và các văn bản liên quan đến SME; tìm kiếm/truy xuất thông tin pháp luật chính xác từ kho dữ liệu; ưu tiên khả năng retrieval và grounding chính xác.
- **Hỏi đáp pháp lý bằng tiếng Việt**: Hiểu ngôn ngữ tự nhiên tiếng Việt, hỏi đáp các tình huống pháp lý thường gặp.
- **Dẫn nguồn điều luật**: Trích dẫn điều/khoản/văn bản liên quan (<mã văn bản>|<tên văn bản>|<điều>). Hạn chế việc trả lời không có căn cứ pháp lý.
- **Tư vấn sơ bộ & cảnh báo giới hạn**: Đưa ra hướng dẫn pháp lý sơ bộ cho người dùng; nhắc nhở rủi ro tuân thủ; hiển thị cảnh báo giới hạn AI.
- **Kiểm soát nội dung sai lệch**: Hạn chế AI sinh thông tin sai lệch; tránh bịa điều luật/nguồn không tồn tại.

---

## 3. Timeline / Deadlines
- **03 tháng 6, 2026**: Ngày khai mạc, phát hành tập dữ liệu kiểm thử
- **30 tháng 6, 2026**: Chính thức đóng cổng hệ thống, các đội phải hoàn thành nộp bài
- **05 tháng 7, 2026**: Công bố kết quả Top 10, tiến vào DemoDay
- **11 tháng 7, 2026**: Ngày DemoDay, công bố kết quả chung cuộc
*Lưu ý: Tất cả các hạn chót đều là 23:59 theo giờ Việt Nam (UTC+07:00).*

---

## 4. Quy định về dữ liệu bên ngoài & Mô hình huấn luyện (PLMs)
- **Dữ liệu bên ngoài**: KHÔNG được sử dụng dữ liệu bên ngoài trong bất kỳ bước xử lý nào.
- **Mô hình ngôn ngữ**:
  - Được sử dụng các mô hình ngôn ngữ huấn luyện trước và LLM có dữ liệu huấn luyện và/hoặc mô hình được công khai có kích thước dưới 14B (ví dụ: Huggingface...).
  - KHÔNG được sử dụng các LLM có mô hình đóng (ví dụ: GPT-4o, Gemini, ...).
  - Chỉ được sử dụng các mô hình được phát hành trước ngày 1 tháng 3 năm 2026 (giờ Việt Nam).
  - Yêu cầu đưa thông tin về cách thức lấy mô hình vào bài báo (working notes paper) phục vụ mục đích tái lập kết quả.

---

## 5. Evaluation Metrics
Hiệu năng hệ thống được đánh giá bằng các chỉ số tự động và thủ công sử dụng trung bình macro.

### 5.1 Truy hồi thông tin (Information Retrieval)
Đánh giá bằng các chỉ số Độ chính xác (Precision), Độ bao phủ (Recall) và điểm F2 macro.
- **Cách trích xuất điều luật từ câu trả lời**: Hệ thống chấm điểm tự động tìm các pattern "Điều X" trong trường `relevant_docs` / `relevant_articles` của bài nộp, so sánh với điều luật trong đáp án (chuẩn hóa về định dạng `Điều X`).
- **Precision**: Trung bình của (số điều luật truy hồi đúng cho mỗi truy vấn) / (số điều luật đã truy hồi cho mỗi truy vấn)
- **Recall**: Trung bình của (số điều luật truy hồi đúng cho mỗi truy vấn) / (số điều luật liên quan của mỗi truy vấn)
- **Chỉ số F2**: $F_2 = \frac{5 \times Precision \times Recall}{4 \times Precision + Recall}$

### 5.2 Hỏi đáp pháp luật (Legal QA)
Bộ tiêu chí đánh giá bao gồm 5 nhóm:
1. **Căn cứ chính xác pháp luật**: Tỷ lệ câu hỏi có ít nhất một điều luật được trích xuất đúng từ câu trả lời. Đánh giá tự động.
2. **Tính chính xác nội dung**: Đánh giá mức độ chính xác của nội dung câu trả lời so với quy định pháp luật.
3. **Tính đầy đủ & toàn diện**: Đánh giá câu trả lời có bao quát đầy đủ các khía cạnh liên quan của câu hỏi không.
4. **Tính thực tiễn – khả năng áp dụng**: Đánh giá câu trả lời có thể áp dụng thực tế trong bối cảnh pháp lý không.
5. **Tính rõ ràng – dễ hiểu**: Đánh giá câu trả lời có diễn đạt rõ ràng, dễ hiểu cho người đọc không chuyên không.

#### 5.2.1 Đánh giá tự động
Sử dụng LLM làm giám khảo tự động (LLM-as-a-Judge) để chấm điểm câu trả lời theo bộ tiêu chí 5 nhóm dựa trên: câu hỏi, câu trả lời tham chiếu, các điều luật căn cứ và câu trả lời của hệ thống.

#### 5.2.2 Con người đánh giá
Một tập con các câu trả lời sẽ được đánh giá độc lập bởi các chuyên gia pháp luật theo cùng bộ tiêu chí 5 nhóm. Điểm Human Evaluation cuối cùng là trung bình cộng của các chuyên gia.
*Lưu ý: 4 chỉ số đánh giá thủ công (Tính chính xác nội dung, Tính đầy đủ & toàn diện, Tính thực tiễn, Tính rõ ràng) hiện được đặt giá trị 0.0 và sẽ được cập nhật điểm số sau khi ban giám khảo hoàn thành đánh giá.*

---

## 6. Data Description & Format
### 6.1 Dữ liệu cung cấp
Ban Tổ chức cung cấp duy nhất bộ dữ liệu kiểm thử (test set). Không cung cấp bất kỳ tập dữ liệu huấn luyện (train) hay tập phát triển (dev) nào.
Đầu vào (câu hỏi):
```json
{
  "id": 1,
  "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"
}
```

### 6.2 Bài nộp lên hệ thống
Định dạng file nộp là một file dự đoán duy nhất tên là `results.json`, sau đó nén thành file `.zip` phẳng (không chứa thư mục con).
Định dạng mẫu của `results.json`:
```json
[
  {
    "id": 1,
    "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?",
    "answer": "Doanh nghiệp được hỗ trợ khi được thành lập, tổ chức và hoạt động theo pháp luật về doanh nghiệp; đáp ứng tiêu chí doanh nghiệp nhỏ và vừa, gồm số lao động tham gia bảo hiểm xã hội bình quân năm không quá 200 người và đáp ứng một trong hai tiêu chí: tổng nguồn vốn không quá 100 tỷ đồng hoặc tổng doanh thu của năm trước liền kề không quá 300 tỷ đồng. Ngoài ra, doanh nghiệp phải đáp ứng điều kiện cụ thể của từng nội dung hỗ trợ và thực hiện đầy đủ nghĩa vụ, trách nhiệm theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa và pháp luật có liên quan.",
    "relevant_docs": [
      "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
      "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
    ],
    "relevant_articles": [
      "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4",
      "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5",
      "80/2021/NĐ-CP|Nghị định 80/2021/NĐ-CP Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5"
    ]
  }
]
```
*Lưu ý về định dạng*:
- `relevant_docs`: `<mã văn bản>|<tên văn bản>`. `<tên văn bản>` viết theo công thức: Loại văn bản + Mã văn bản + Trích yếu.
- `relevant_articles`: `<mã văn bản>|<tên văn bản>|<điều>`. `<tên văn bản>` viết theo công thức: Loại văn bản + Mã văn bản + Trích yếu.

### 6.3 Hướng dẫn tạo file nộp bài
Nén file bằng lệnh:
- **Linux/macOS**: `zip submission.zip results.json`
- **Windows (PowerShell)**: `Compress-Archive -Path results.json -DestinationPath submission.zip`
Tải lên tại mục **My Submissions** trên `http://leaderboard.aiguru.com.vn/`.

---

## 7. Submission Rules
- **Số lượt nộp**: Tối đa 10 bài/ngày.
- **Số bài nộp tối đa trong Vòng Riêng (Private Phase)**: 5 bài tổng cộng.
- **Chọn bài chấm QA**: Phải chọn một bài trong các bài đã nộp và đẩy (promote) lên bảng xếp hạng. Ban tổ chức sẽ đánh giá QA cho các bài trên bảng xếp hạng định kỳ mỗi tuần một lần.
- **Tên đội**: Đặt tên người dùng đại diện cho đội.
- **Bài báo**: Kết quả cuối cùng sẽ chỉ được công nhận sau khi nộp bài báo mô tả phương pháp (working notes paper).

---

## 8. Ban Tổ chức liên hệ
- **AI Guru** – Công ty Cổ phần Tập đoàn Dagoras Group
- **Địa chỉ**: Tầng 8, số 80 Duy Tân, Cầu Giấy, Hà Nội
- **Đầu mối liên hệ**:
  - Nguyễn Thị Minh Nguyệt, ĐT: 0981544974, Email: nguyetntm@dagoras.io
  - Vũ Thị Thuỳ Linh, ĐT: 0961891198, Email: linhvtt@dagoras.io

