**BỘ KHOA HỌC VÀ CÔNG NGHỆ**
**HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG**
**CƠ SỞ TẠI THÀNH PHỐ HỒ CHÍ MINH**
*******

**BÁO CÁO ĐỒ ÁN**
**KHAI PHÁ DỮ LIỆU ĐA PHƯƠNG TIỆN**

**ĐỀ TÀI: **SỬ DỤNG PHƯƠNG PHÁP TƯƠNG QUAN MÀU SẮC (COLOR CORRELOGRAMS) CHO BÀI TOÁN NHẬN DẠNG ẢNH BẰNG PHƯƠNG PHÁP HỌC MÁY

**GIẢNG VIÊN HƯỚNG DẪN: **NGUYỄN NGỌC DUY

**SINH VIÊN THỰC HIỆN**
**NHÓM 1****1**

TRẦN QUANG HÙNG     N22DCPT035
VŨ QUANG LONG           N22DCPT052

**TPHCM, ngày ****22**** tháng**** 4**** năm 202****6**
**LỜI CẢM ƠN**
Trong quá trình thực hiện đề tài *"Sử dụng phương pháp tương quan màu sắc (Color Correlograms) cho bài toán nhận dạng ảnh bằng phương pháp học máy"* thuộc môn khai phá dữ liệu đa phương tiện, chúng em đã nhận được nhiều sự hỗ trợ quý báu từ thầy Nguyễn Ngọc Duy, bạn bè và những nguồn tài liệu chuyên môn liên quan đến lĩnh vực xử lý ảnh, học máy và khai phá dữ liệu. Đây là một đề tài vừa mang tính kỹ thuật, vừa đòi hỏi khả năng phân tích dữ liệu, xây dựng mô hình và hiện thực hóa thành một hệ thống nhận dạng ảnh hiệu quả.

Đặc biệt, chúng em xin gửi lời cảm ơn chân thành và sâu sắc nhất đến thầy Nguyễn Ngọc Duy. Thầy không chỉ là người đã truyền đạt cho chúng em các kiến thức nền tảng về lập trình, xử lý dữ liệu đa phương tiện, học máy và phương pháp nghiên cứu, mà còn là giảng viên trực tiếp hướng dẫn đồ án. Thầy đã tận tình chỉ bảo, định hướng cách tiếp cận đề tài, góp ý về nội dung triển khai cũng như hỗ trợ chúng em hệ thống hóa kiến thức thành một báo cáo có cấu trúc chặt chẽ. Những nền tảng và sự hướng dẫn của thầy chính là cơ sở vững chắc để chúng em có thể phân tích, xây dựng và hoàn thiện hệ thống nhận dạng ảnh này một cách hiệu quả.

Bên cạnh đó, chúng em xin cảm ơn bạn bè và các anh chị đã chia sẻ kinh nghiệm, tài liệu tham khảo và ý kiến đóng góp trong quá trình hoàn thiện sản phẩm. Những trao đổi thực tế về xử lý dữ liệu ảnh, xây dựng đặc trưng, áp dụng phương pháp Color Correlograms và triển khai mô hình học máy đã hỗ trợ chúng em rất nhiều.

Mặc dù đã cố gắng hoàn thành báo cáo và sản phẩm với tinh thần nghiêm túc, nhưng do giới hạn về thời gian, kinh nghiệm và phạm vi nghiên cứu, nội dung báo cáo chắc chắn vẫn còn một số thiếu sót nhất định. Chúng em rất mong tiếp tục nhận được sự nhận xét và góp ý từ thầy Nguyễn Ngọc Duy để đề tài được hoàn thiện hơn trong các lần phát triển tiếp theo.

Nhóm chúng em xin chân thành cảm ơn!

**NHẬN XÉT CỦA GIẢNG VIÊN**

......................................................................................................................
......................................................................................................................
......................................................................................................................
......................................................................................................................
......................................................................................................................
......................................................................................................................
......................................................................................................................
......................................................................................................................
......................................................................................................................
......................................................................................................................

**Điểm: **........................................

**Giảng viên ký tên**

[Chữ ký và họ tên]

**DANH MỤC ****BẢNG VÀ HÌNH ẢNH**
Bảng 2.1. So sánh Color Histogram và Color Correlogram15
Hình 2.1. Các chỉ số đánh giá mô hình18
Hình 3.1. Pipeline tổng quát của hệ thống nhận dạng ảnh19
Bảng 3.1. Danh sách các lớp ảnh trong bộ dữ liệu Corel-1K20
Bảng 3.2. Kích thước các ma trận đặc trưng đã trích xuất22
Bảng 3.3. Mô tả chức năng các tệp mã nguồn chính24
Hình 3.2. Giao diện tab Nhận dạng ảnh trong ứng dụng Streamlit26
Hình 3.3. Màn hình kết quả trực tiếp sau khi dự đoán và truy hồi ảnh tương tự27
Hình 3.4. Giao diện tab Thực nghiệm thuật toán của ứng dụng Streamlit28
Hình 3.5. Bảng điều khiển kết quả và đường dẫn artifact sau khi chạy thí nghiệm29
Hình 3.6. Bảng báo cáo phân loại theo lớp29
Hình 3.7. Bảng báo ma trận nhầm lẫn30
Bảng 4.1. Không gian tham số tìm kiếm GridSearchCV cho từng mô hình32
Bảng 4.2. Kết quả huấn luyện - Train CV Accuracy và tham số tốt nhất32
Bảng 4.3. Kết quả đánh giá trên tập kiểm tra độc lập (200 ảnh test)33
Bảng 4.4. So sánh Train CV Accuracy và Test Accuracy giữa các cấu hình35

**MỤC LỤC**

**LỜI CẢM ƠN**1
**NHẬN XÉT CỦA GIẢNG VIÊN**2
**DANH MỤC BẢNG VÀ HÌNH ẢNH**3
**MỤC LỤC**4
**LỜI MỞ ĐẦU**7
**CHƯƠNG 1. TỔNG QUAN ĐỀ TÀI**9
**1.1. Bối cảnh của bài toán nhận dạng ảnh**9
**1.2. Lý do chọn đề tài**9
**1.3. Mục tiêu nghiên cứu**10
**1.4. Đối tượng và phạm vi nghiên cứu**11
**1.5. Phương pháp nghiên cứu**11
**1.6. Ý nghĩa khoa học và thực tiễn**11
**1.7. Cấu trúc của báo cáo**12
**CHƯƠNG 2. CƠ SỞ LÝ THUYẾT**13
**2.1. Khái quát về bài toán nhận dạng ảnh**13
**2.2. Không gian màu RGB và HSV**13
**2.3. Color Histogram**14
**2.4. Color Correlogram**14
***2.4.1. Khái niệm***14
***2.4.2. Công thức của Auto-Correlogram***15
***2.4.3. So sánh Color Correlogram và Color Histogram***15
**2.5. Spatial Color Correlogram**16
**2.6. Các mô hình học máy sử dụng trong đề tài**16
***2.6.1. Support Vector Machine (SVM)***16
***2.6.2. K-Nearest Neighbors (KNN)***16
***2.6.3. Phạm vi mô hình trong triển khai hiện tại***17
**2.7. Các chỉ số đánh giá mô hình**17
**2.8. Chia tập dữ liệu và tránh rò rỉ dữ liệu**18
**CHƯƠNG 3. THIẾT KẾ VÀ XÂY DỰNG HỆ THỐNG**19
**3.1. Tổng quan kiến trúc hệ thống**19
**3.2. Bộ dữ liệu Corel-1K**19
**3.3. Tiền xử lý dữ liệu ảnh**20
**3.4. Lượng tử hóa màu**21
**3.5. Thiết kế đặc trưng**21
***3.5.1. Histogram màu***21
***3.5.2. Auto-Correlogram***21
***3.5.3. Spatial Correlogram***22
**3.6. Tổ chức dữ liệu đặc trưng**22
**3.7. Chia tập dữ liệu**23
**3.8. Thiết kế pipeline huấn luyện**23
**3.9. Cấu trúc mã nguồn dự án**23
**3.10. Ứng dụng minh họa bằng Streamlit**25
***3.10.1. Tab nhận dạng ảnh***25
***3.10.2. Kết quả trực tiếp và truy hồi ảnh tương tự***26
***3.10.3. Tab thực nghiệm thuật toán***27
***3.10.4. Kết quả thí nghiệm và artifact đã lưu***28
**CHƯƠNG 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ**31
**4.1. Mục tiêu thực nghiệm**31
**4.2. Môi trường thực nghiệm**31
**4.3. Quy trình thực nghiệm**31
**4.4. Các cấu hình tham số được tìm kiếm**32
**4.5. Kết quả huấn luyện và cross-validation trên train split**32
**4.6. Kết quả đánh giá trên tập kiểm tra độc lập**33
**4.7. Phân tích kết quả**34
***4.7.1. So sánh giữa Histogram và Correlogram***34
***4.7.2. So sánh giữa không gian màu HSV và RGB***34
***4.7.3. So sánh giữa các mô hình học máy***35
***4.7.4. Vai trò của Spatial Correlogram***35
**4.8. Phân tích chênh lệch giữa Train CV và Test**35
**4.9. Tổng hợp kết quả và nhận xét chung**37
**4.10. Hạn chế của thực nghiệm**37
**CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN**38
**5.1. Kết luận**38
**5.2. Hướng phát triển**39
**TÀI LIỆU THAM KHẢO**40

**LỜI MỞ ĐẦU**
Trong bối cảnh dữ liệu hình ảnh ngày càng xuất hiện với quy mô lớn trong nhiều lĩnh vực như truyền thông số, thương mại điện tử, y tế, giám sát và lưu trữ đa phương tiện, nhu cầu xây dựng các hệ thống có khả năng nhận dạng và phân loại ảnh tự động ngày càng trở nên quan trọng. Bài toán nhận dạng ảnh không chỉ có ý nghĩa về mặt học thuật mà còn mang giá trị ứng dụng cao trong thực tiễn, đặc biệt khi cần xử lý một lượng lớn ảnh mà con người khó có thể phân loại thủ công một cách nhanh chóng và nhất quán.

Trong nhiều hướng tiếp cận của bài toán nhận dạng ảnh, việc biểu diễn ảnh dưới dạng đặc trưng là bước đóng vai trò nền tảng. Nếu đặc trưng trích xuất được thể hiện tốt bản chất thị giác của ảnh thì các mô hình học máy phía sau mới có điều kiện hoạt động hiệu quả. Một trong những nhóm đặc trưng cổ điển quan trọng là đặc trưng màu sắc. Màu sắc phản ánh mạnh mẽ nội dung thị giác tổng quát của ảnh, đặc biệt đối với các bộ dữ liệu cảnh thiên nhiên, động vật, thực phẩm hoặc đối tượng có phân bố màu đặc trưng. Tuy nhiên, nếu chỉ sử dụng histogram màu, hệ thống mới chỉ nắm được tần suất xuất hiện của màu mà chưa mô tả được mối quan hệ không gian giữa các màu trong ảnh.

Để giải quyết hạn chế đó, phương pháp Color Correlogram đã được đề xuất như một cách mô tả giàu thông tin hơn. Thay vì chỉ đếm xem màu nào xuất hiện nhiều hay ít, Color Correlogram còn phản ánh xác suất để hai điểm ảnh có cùng màu xuất hiện ở một khoảng cách nhất định. Nhờ vậy, đặc trưng này vừa giữ được thông tin màu, vừa bổ sung được một phần cấu trúc không gian, giúp tăng khả năng phân biệt giữa các lớp ảnh có histogram tương tự nhau nhưng bố cục màu khác nhau.

Tiểu luận này tập trung nghiên cứu việc sử dụng Color Correlogram cho bài toán nhận dạng ảnh bằng phương pháp học máy. Nội dung báo cáo kết hợp giữa phần cơ sở lý thuyết và phần thực nghiệm trên bộ dữ liệu Corel-1K. Hệ thống được xây dựng với quy trình đầy đủ gồm tiền xử lý ảnh, lượng tử hóa màu, trích xuất đặc trưng Color Correlogram và Color Histogram, huấn luyện các mô hình học máy như SVM và KNN, đánh giá trên tập kiểm tra độc lập, đồng thời xây dựng một ứng dụng minh họa để thử nghiệm dự đoán ảnh đầu vào.
Điểm nhấn của đề tài không nằm ở việc theo đuổi các mô hình học sâu hiện đại, mà ở việc nghiên cứu một hướng tiếp cận cổ điển nhưng có tính giải thích tốt, dễ phân tích, dễ triển khai và phù hợp với quy mô dữ liệu vừa phải. Qua đó, đề tài cho thấy rằng với một đặc trưng được thiết kế hợp lý như Color Correlogram, kết hợp cùng các mô hình học máy truyền thống, hệ thống vẫn có thể đạt kết quả nhận dạng ảnh khả quan và mang ý nghĩa học thuật rõ ràng.

**CHƯƠNG 1. TỔNG QUAN ĐỀ TÀI**
**1.1. Bối cảnh của bài toán nhận dạng ảnh**
Nhận dạng ảnh là một bài toán trung tâm trong lĩnh vực thị giác máy tính và khai phá dữ liệu đa phương tiện. Mục tiêu của bài toán là xây dựng hệ thống có khả năng tiếp nhận một ảnh đầu vào, phân tích nội dung thị giác của ảnh đó và gán nó vào một lớp phù hợp trong tập lớp đã được định nghĩa trước. Trong thực tế, bài toán này xuất hiện trong nhiều tình huống như phân loại ảnh phong cảnh, xác định loại đối tượng, tìm kiếm ảnh theo nội dung, hỗ trợ tổ chức kho ảnh số hoặc xây dựng hệ thống gợi ý trực quan.
Về bản chất, ảnh số là dữ liệu có chiều rất cao. Một ảnh màu kích thước 128x128 đã có tới hàng chục nghìn giá trị điểm ảnh. Nếu sử dụng trực tiếp toàn bộ giá trị điểm ảnh để huấn luyện mô hình học máy thì hệ thống dễ gặp khó khăn do số chiều lớn, nhiễu cao và tính phụ thuộc mạnh vào vị trí cụ thể của từng điểm ảnh. Vì vậy, cần có bước trích xuất đặc trưng nhằm chuyển đổi ảnh từ dạng dữ liệu thô sang một vector đặc trưng gọn hơn nhưng vẫn giữ được thông tin quan trọng phục vụ phân biệt lớp.
Trong các đặc trưng ảnh cổ điển, đặc trưng màu đóng vai trò rất quan trọng. Với nhiều bộ dữ liệu như cảnh biển, núi, hoa, thực phẩm, động vật hoặc phương tiện giao thông, phân bố màu là một tín hiệu rất mạnh để phân biệt ảnh giữa các lớp. Tuy nhiên, không phải mọi đặc trưng màu đều hiệu quả như nhau. Histogram màu là cách mô tả đơn giản và phổ biến, nhưng nó bỏ qua hoàn toàn mối quan hệ không gian giữa các màu. Điều này dẫn đến khả năng hai ảnh có bố cục rất khác nhau nhưng lại có histogram gần giống nhau.
Chính từ hạn chế đó, các phương pháp biểu diễn đặc trưng có bổ sung thông tin không gian màu được quan tâm, trong đó Color Correlogram là một đại diện tiêu biểu. Phương pháp này không chỉ mô tả màu gì xuất hiện mà còn mô tả màu đó phân bố với quan hệ không gian như thế nào trong ảnh.
**1.2. Lý do chọn đề tài**
Việc lựa chọn đề tài "Sử dụng phương pháp Tương quan màu sắc (Color Correlograms) cho bài toán nhận dạng ảnh bằng phương pháp học máy" xuất phát từ cả góc độ học thuật lẫn thực tiễn triển khai.
Thứ nhất, Color Correlogram là một đặc trưng có giá trị học thuật vì nó nằm ở giao điểm giữa xử lý ảnh và học máy. Đặc trưng này giúp người học tiếp cận bài toán nhận dạng ảnh theo hướng có thể giải thích được, thay vì chỉ sử dụng mô hình như một "hộp đen". Khi nghiên cứu Color Correlogram, người thực hiện phải hiểu rõ ý nghĩa của màu sắc, cách lượng tử hóa màu, vai trò của khoảng cách không gian và ảnh hưởng của biểu diễn đặc trưng tới hiệu quả mô hình phân loại.
Thứ hai, đây là một hướng tiếp cận phù hợp với điều kiện học tập và nghiên cứu ở quy mô học phần. So với các mô hình học sâu đòi hỏi dữ liệu lớn, tài nguyên tính toán mạnh và thời gian huấn luyện dài, việc kết hợp Color Correlogram với các mô hình học máy truyền thống như SVM và KNN giúp quá trình triển khai khả thi hơn.
Thứ ba, đề tài mang ý nghĩa so sánh phương pháp rõ ràng. Bên cạnh Color Correlogram, hệ thống còn cài đặt Color Histogram làm baseline. Nhờ đó, bài toán không chỉ dừng ở việc xây dựng một hệ thống nhận dạng ảnh, mà còn trả lời được câu hỏi nghiên cứu quan trọng: việc bổ sung thông tin tương quan không gian của màu sắc có thực sự cải thiện hiệu quả phân loại so với đặc trưng màu đơn giản hay không.
Cuối cùng, đề tài có khả năng mở rộng tốt. Dù trong phạm vi hiện tại, hệ thống mới áp dụng cho bộ dữ liệu Corel-1K và các mô hình học máy cổ điển, nhưng kết quả nghiên cứu có thể làm nền tảng cho các hướng phát triển tiếp theo.
**1.3. Mục tiêu nghiên cứu**
Mục tiêu tổng quát của đề tài là nghiên cứu khả năng sử dụng đặc trưng Color Correlogram trong bài toán nhận dạng ảnh và đánh giá hiệu quả của đặc trưng này khi kết hợp với các mô hình học máy truyền thống.
Để đạt được mục tiêu tổng quát đó, đề tài triển khai các mục tiêu cụ thể sau:
Tìm hiểu cơ sở lý thuyết của Color Correlogram, bao gồm cách biểu diễn mối quan hệ màu sắc theo khoảng cách không gian, sự khác biệt giữa Color Correlogram và Color Histogram, cũng như vai trò của lượng tử hóa màu trong quá trình trích xuất đặc trưng.
Xây dựng pipeline hoàn chỉnh cho bài toán nhận dạng ảnh, bao gồm đọc dữ liệu, chuẩn hóa kích thước ảnh, trích xuất đặc trưng, chia tập dữ liệu, huấn luyện mô hình và đánh giá kết quả.
Tiến hành thực nghiệm trên bộ dữ liệu Corel-1K với nhiều tổ hợp mô hình khác nhau để so sánh hiệu quả giữa các đặc trưng và thuật toán học máy.
Xây dựng ứng dụng minh họa để hỗ trợ dự đoán ảnh đầu vào và hiển thị các ảnh tương tự trong cơ sở dữ liệu.
**1.4. Đối tượng và phạm vi nghiên cứu**
Đối tượng nghiên cứu chính của đề tài là đặc trưng Color Correlogram và việc áp dụng đặc trưng này vào bài toán phân loại ảnh bằng phương pháp học máy có giám sát. Ngoài ra, đề tài cũng khảo sát các mô hình học máy phổ biến gồm Support Vector Machine và K-Nearest Neighbors.
Về phạm vi dữ liệu, đề tài sử dụng bộ dữ liệu Corel-1K gồm 1000 ảnh thuộc 10 lớp khác nhau. Đây là một bộ dữ liệu kinh điển, có kích thước vừa phải và đủ đa dạng để kiểm tra khả năng phân loại của hệ thống. Các lớp dữ liệu bao gồm africans, beaches, buildings, buses, dinosaurs, elephants, flowers, food, horses và mountains.
Về phạm vi phương pháp, đề tài tập trung vào đặc trưng màu và các mô hình học máy truyền thống. Đề tài không đi sâu vào mạng nơ-ron tích chập hay các mô hình học sâu khác.
**1.5. Phương pháp nghiên cứu**
Đề tài sử dụng kết hợp phương pháp nghiên cứu lý thuyết và thực nghiệm. Ở phần lý thuyết, báo cáo tổng hợp các khái niệm nền tảng liên quan đến nhận dạng ảnh, đặc trưng màu, Color Histogram, Color Correlogram và các mô hình học máy sử dụng trong hệ thống.
Ở phần thực nghiệm, đề tài xây dựng hệ thống bằng ngôn ngữ Python cùng các thư viện phổ biến như OpenCV, NumPy, scikit-learn, matplotlib, seaborn, joblib và Streamlit. Dữ liệu được chuẩn hóa kích thước, trích xuất đặc trưng theo nhiều cấu hình khác nhau, sau đó huấn luyện và đánh giá bằng quy trình chia tập train/test cố định. Việc chọn tham số được thực hiện bằng StratifiedKFold trên train split, còn tập test được giữ riêng hoàn toàn cho đánh giá cuối. Các chỉ số đánh giá như Accuracy, Precision, Recall và F1-score được sử dụng để phân tích hiệu quả mô hình một cách toàn diện.
**1.6. Ý nghĩa khoa học và thực tiễn**
Về ý nghĩa khoa học, đề tài góp phần minh họa rõ mối liên hệ giữa biểu diễn đặc trưng và hiệu quả phân loại trong nhận dạng ảnh. Thông qua Color Correlogram, đề tài cho thấy rằng việc thêm thông tin không gian vào biểu diễn màu có thể tạo khác biệt đáng kể so với các đặc trưng chỉ dựa trên tần suất xuất hiện màu.
Về ý nghĩa thực tiễn, kết quả nghiên cứu cho thấy hoàn toàn có thể xây dựng một hệ thống nhận dạng ảnh có độ chính xác tương đối tốt mà không cần đến các mô hình học sâu phức tạp. Với các bộ dữ liệu cỡ vừa và tài nguyên tính toán giới hạn, cách tiếp cận này vẫn có giá trị ứng dụng trong môi trường học tập, nghiên cứu và một số bài toán thực tế có quy mô nhỏ đến trung bình.
**1.7. Cấu trúc của báo cáo**
Báo cáo được tổ chức thành năm chương chính. Chương 1 trình bày tổng quan về đề tài, bối cảnh, mục tiêu, phạm vi và phương pháp nghiên cứu. Chương 2 tập trung vào cơ sở lý thuyết liên quan đến nhận dạng ảnh, đặc trưng màu, Color Histogram, Color Correlogram, các mô hình học máy và các chỉ số đánh giá. Chương 3 mô tả thiết kế và xây dựng hệ thống. Chương 4 trình bày kết quả thực nghiệm và phân tích hiệu năng mô hình. Cuối cùng, Chương 5 đưa ra kết luận và đề xuất hướng phát triển.

**CHƯƠNG 2. CƠ SỞ LÝ THUYẾT**
**2.1. Khái quát về bài toán nhận dạng ảnh**
Nhận dạng ảnh là bài toán gán một ảnh đầu vào vào một trong các lớp đã biết từ trước. Trong bối cảnh học máy có giám sát, điều này tương ứng với việc xây dựng một hàm ánh xạ từ không gian ảnh sang không gian nhãn lớp dựa trên tập dữ liệu huấn luyện đã gán nhãn. Quá trình này thường trải qua ba bước chính: biểu diễn ảnh thành đặc trưng, huấn luyện mô hình từ đặc trưng và nhãn, sau đó sử dụng mô hình đã học để dự đoán cho dữ liệu mới.
Một thách thức quan trọng của nhận dạng ảnh là ảnh chứa lượng thông tin rất lớn nhưng không phải thông tin nào cũng hữu ích cho phân loại. Các yếu tố như kích thước ảnh, ánh sáng, góc nhìn, nền, nhiễu hoặc biến đổi cục bộ đều có thể làm giảm hiệu quả mô hình nếu ảnh không được biểu diễn phù hợp. Do đó, bước trích xuất đặc trưng có vai trò then chốt.
Các loại đặc trưng ảnh truyền thống có thể kể đến gồm đặc trưng màu, đặc trưng kết cấu, đặc trưng hình dạng và đặc trưng cục bộ. Trong phạm vi đề tài này, trọng tâm được đặt vào đặc trưng màu, bởi bộ dữ liệu sử dụng gồm các lớp có sự khác biệt tương đối rõ về gam màu chủ đạo và phân bố màu trong ảnh.
**2.2. Không gian màu RGB và HSV**
Không gian màu RGB là không gian màu cộng, trong đó một màu được xác định bởi cường độ của ba kênh đỏ, xanh lá và xanh dương. Đây là không gian gắn chặt với cách hiển thị ảnh trên thiết bị số, vì vậy hầu hết ảnh đầu vào đều ở dạng RGB hoặc BGR. Ưu điểm của RGB là dễ sử dụng, trực tiếp và ít tốn chi phí chuyển đổi. Tuy nhiên, các kênh màu trong RGB thường tương quan với nhau và không phản ánh trực quan tốt theo cảm nhận màu của con người [4].
Không gian màu HSV gồm ba thành phần: Hue (H) mô tả loại màu, Saturation (S) mô tả độ đậm nhạt của màu và Value (V) mô tả độ sáng. Khi chuyển sang HSV, màu sắc được phân tách rõ ràng hơn theo bản chất cảm nhận. Trong nhiều trường hợp, biểu diễn này phù hợp hơn cho việc trích xuất đặc trưng màu và giúp mô hình ổn định hơn khi điều kiện sáng thay đổi ở mức vừa phải.
Trong đề tài này, ảnh được lượng tử hóa theo HSV với cấu hình H=8, S=3, V=3, tạo ra tổng cộng 8 × 3 × 3 = 72 màu. Đối với RGB, mỗi kênh được chia thành 4 mức, tạo thành 4³ = 64 màu. Việc chọn số bin như vậy là một thỏa hiệp giữa khả năng mô tả và độ phức tạp của đặc trưng.
**2.3. Color Histogram**
Color Histogram là một trong những cách biểu diễn màu phổ biến nhất trong xử lý ảnh. Ý tưởng cơ bản là đếm số điểm ảnh thuộc mỗi màu sau khi lượng tử hóa, sau đó chuẩn hóa để tạo thành một vector biểu diễn phân bố màu trong ảnh. Nếu gọi số lượng màu sau lượng tử hóa là n, thì histogram sẽ là một vector n chiều, trong đó mỗi phần tử phản ánh tần suất xuất hiện của một màu tương ứng [4].
Ưu điểm lớn nhất của Color Histogram là đơn giản, dễ cài đặt, tốc độ tính toán nhanh và có tính khái quát tốt ở mức toàn cục. Đặc trưng này thường hoạt động hiệu quả khi các lớp ảnh khác nhau có phân bố màu tương đối khác biệt [4].
Tuy nhiên, hạn chế quan trọng của Color Histogram là hoàn toàn không mang thông tin không gian. Hai ảnh có thể có cùng tỷ lệ màu nhưng bố cục màu khác nhau hoàn toàn vẫn cho histogram gần giống nhau. Chẳng hạn, một ảnh bãi biển với dải trời xanh phía trên và cát vàng phía dưới có thể có histogram tương tự một ảnh khác trong đó vùng màu xanh và vàng được trộn lẫn theo bố cục khác. Chính vì lý do đó, Color Histogram được sử dụng trong đề tài như một baseline để so sánh với Color Correlogram [4].
**2.4. Color Correlogram**
***2.4.1. Khái niệm***
Color Correlogram là một đặc trưng mô tả sự tương quan không gian giữa các màu trong ảnh. Theo định nghĩa kinh điển, correlogram biểu diễn xác suất để tại một khoảng cách d nhất định, một điểm ảnh có màu ci sẽ có một điểm ảnh khác mang màu cj. Trong trường hợp đơn giản hơn, nếu chỉ xét xác suất giữa các điểm ảnh cùng màu, ta thu được Auto-Correlogram.
Khác với histogram chỉ trả lời câu hỏi "màu nào xuất hiện bao nhiêu", Color Correlogram trả lời câu hỏi "màu nào không chỉ xuất hiện, mà còn được tổ chức trong không gian ảnh như thế nào". Đây chính là phần thông tin bổ sung làm cho đặc trưng này mạnh hơn trong nhiều tình huống [1].
***2.4.2. Công thức của Auto-Correlogram***
Giả sử ảnh sau khi lượng tử hóa có tập màu C, và Ic là tập các điểm ảnh có màu c. Auto-Correlogram tại khoảng cách d được biểu diễn như sau:
**α(c, d) = Pr[p₂ ∈ Ic | p₁ ∈ Ic, |p₁ - p₂| = d]**
Trong đó, p₁ và p₂ là hai điểm ảnh, còn |p₁ - p₂| = d biểu thị rằng hai điểm này cách nhau một khoảng cách xác định. Công thức cho thấy với mỗi màu c và mỗi khoảng cách d, ta ước lượng xác suất một điểm ảnh cùng màu xuất hiện ở lân cận.
Trong cài đặt của đề tài, các khoảng cách được chọn là {1, 3, 5, 7}. Đây là các giá trị đủ để phản ánh cả tương quan cục bộ gần và tương quan ở phạm vi xa hơn trong ảnh. Với cấu hình HSV gồm 72 màu và 4 khoảng cách, đặc trưng Correlogram toàn ảnh có số chiều là 72 × 4 = 288. Với RGB gồm 64 màu và 4 khoảng cách, số chiều là 64 × 4 = 256 [1].
***2.4.3. So sánh Color Correlogram và Color Histogram***
So với Color Histogram, Color Correlogram có chi phí tính toán cao hơn và vector đặc trưng dài hơn. Tuy nhiên, đổi lại, nó cung cấp thông tin giàu hơn đáng kể. Histogram chỉ là mô tả phân bố màu bậc một, còn Correlogram có thể xem như một mô tả mức cao hơn, phản ánh quan hệ không gian của màu trong ảnh. Bảng 2.1 dưới đây so sánh đặc điểm của hai phương pháp đặc trưng màu [1][4].
Bảng 2.1. So sánh Color Histogram và Color Correlogram
| **Tiêu chí** | **Color Histogram** | **Color Correlogram** |
| --- | --- | --- |
| Thông tin màu | Có (tần suất) | Có (tần suất + tương quan) |
| Thông tin không gian | Không | Có (theo khoảng cách d) |
| Số chiều (HSV 72 màu) | 72 chiều | 288 chiều (4 khoảng cách) |
| Tốc độ tính toán | Rất nhanh | Chậm hơn |
| Khả năng phân biệt lớp | Trung bình | Cao hơn |

Như thể hiện trong Bảng 2.1, Color Correlogram đổi chi phí tính toán lấy khả năng phân biệt lớp cao hơn nhờ bổ sung thêm chiều thông tin về quan hệ không gian giữa các màu.
**2.5. Spatial Color Correlogram**
Một mở rộng hợp lý của Color Correlogram là bổ sung thông tin bố cục không gian ở mức vùng ảnh. Thay vì chỉ tính Correlogram trên toàn ảnh, ta có thể chia ảnh thành một số ô lưới và tính đặc trưng cho từng ô. Trong hệ thống của đề tài, ảnh được chia theo lưới 2×2, đồng thời vẫn giữ thêm Correlogram toàn ảnh.
Với cấu hình HSV có 72 màu, 4 khoảng cách và lưới 2×2 kèm đặc trưng toàn ảnh, số chiều của Spatial Correlogram là:
**(1 + 2 × 2) × 72 × 4 = 1440 chiều**
Cách làm này có ý nghĩa vì nhiều lớp ảnh không chỉ khác nhau ở màu mà còn khác nhau ở bố cục tổng quát. Ví dụ, ảnh phong cảnh biển thường có vùng trời phía trên và nước hoặc cát phía dưới; ảnh hoa có thể tập trung màu sắc nổi bật ở vùng trung tâm; ảnh xe buýt có thể chứa khối màu đậm ở vùng giữa ảnh. Kết quả thực nghiệm cho thấy đây là đặc trưng hiệu quả nhất trong toàn bộ các cấu hình được thử nghiệm.
**2.6. Các mô hình học máy sử dụng trong đề tài**
***2.6.1. Support Vector Machine (SVM)***
Support Vector Machine là một trong những thuật toán phân loại mạnh mẽ nhất trong học máy truyền thống, đặc biệt hiệu quả với dữ liệu đặc trưng có chiều tương đối cao. Ý tưởng cốt lõi của SVM là tìm một siêu phẳng phân cách các lớp sao cho biên phân cách giữa chúng là lớn nhất. Trong trường hợp dữ liệu không phân tách tuyến tính, SVM có thể sử dụng các kernel để ánh xạ dữ liệu sang không gian đặc trưng cao hơn.
Trong bài toán của đề tài, SVM phù hợp vì vector Color Correlogram có chiều khá lớn, đặc biệt là Spatial Correlogram với 1440 chiều. SVM kết hợp với chuẩn hóa dữ liệu bằng StandardScaler giúp giảm ảnh hưởng của sự khác biệt về thang đo giữa các chiều. Các tham số quan trọng gồm C, gamma và loại kernel, được lựa chọn bằng GridSearchCV [2][5].

***2.6.2. K-Nearest Neighbors (KNN)***
K-Nearest Neighbors là thuật toán phân loại dựa trên nguyên lý lân cận gần nhất. Khi dự đoán một mẫu mới, thuật toán tìm k điểm gần nhất trong tập huấn luyện theo một độ đo khoảng cách, sau đó bỏ phiếu để quyết định nhãn lớp. Ưu điểm của KNN là không đòi hỏi giai đoạn huấn luyện phức tạp. Tuy nhiên, KNN nhạy cảm với thang đo dữ liệu và thường bị ảnh hưởng khi số chiều đặc trưng tăng cao, do hiện tượng "lời nguyền số chiều" [3].
***2.6.3. Phạm vi mô hình trong triển khai hiện tại***
Trong phiên bản chính thức của dự án, nhóm tập trung vào hai họ mô hình là Support Vector Machine và K-Nearest Neighbors. Đây là các mô hình đã được tích hợp đầy đủ trong pipeline huấn luyện, đánh giá và ứng dụng minh họa; nhờ đó toàn bộ mã nguồn, kết quả thực nghiệm và tài liệu có thể được duy trì nhất quán.
**2.7. Các chỉ số đánh giá mô hình**
Đánh giá mô hình là bước không thể thiếu để xác định chất lượng thực tế của hệ thống. Trong bài toán phân loại nhiều lớp, nếu chỉ nhìn vào Accuracy thì chưa đủ, đặc biệt khi muốn phân tích sâu hơn sự cân bằng giữa các lớp. Đề tài sử dụng bốn chỉ số chính theo cách tiếp cận quen thuộc trong scikit-learn [5]:
Accuracy là tỷ lệ dự đoán đúng trên tổng số mẫu. Đây là chỉ số trực quan và dễ hiểu nhất, phản ánh chất lượng tổng quát của mô hình.
Precision phản ánh mức độ chính xác của các dự đoán dương theo từng lớp. Đề tài sử dụng macro average, tức là tính precision riêng cho từng lớp rồi lấy trung bình.
Recall phản ánh khả năng phát hiện đúng các mẫu thực sự thuộc từng lớp. Nếu recall thấp ở một lớp nào đó, điều này có nghĩa là mô hình thường bỏ sót lớp đó.
F1-score là trung bình điều hòa giữa precision và recall, phù hợp khi cần đánh giá tổng hợp giữa độ chính xác và khả năng bao phủ. Macro F1-score được dùng để tránh thiên lệch theo lớp có số lượng mẫu lớn hơn.
Ngoài ra, Confusion Matrix (ma trận nhầm lẫn) cho biết mỗi lớp thực sự được mô hình dự đoán thành các lớp nào. Đây là công cụ rất hữu ích để phân tích sai số chi tiết theo từng lớp.

Hình 2.1. Các chỉ số đánh giá mô hình
Hình 2.1 nhấn mạnh rằng Accuracy chỉ phản ánh số dự đoán đúng tổng quát, còn Precision, Recall và F1-score cho phép đọc rõ hơn mức cân bằng giữa các lớp. Trong bài toán Corel-1K có 10 lớp, việc báo cáo đồng thời các chỉ số macro và ma trận nhầm lẫn giúp phần đánh giá tránh bị lệch theo một lớp riêng lẻ [5].
**2.8. Chia tập dữ liệu và tránh rò rỉ dữ liệu**
Một vấn đề rất quan trọng trong thực nghiệm học máy là tránh rò rỉ dữ liệu (data leakage). Nếu dữ liệu kiểm tra bị sử dụng gián tiếp trong quá trình lựa chọn tham số hoặc huấn luyện mô hình, kết quả đánh giá sẽ trở nên quá lạc quan và không còn phản ánh đúng năng lực khái quát hóa của hệ thống.
Trong phiên bản hiện tại của đề tài, dữ liệu được chia thành hai phần: train (80%) và test (20%) theo metadata cố định và được phân tầng theo lớp. Tập train dùng để huấn luyện và tìm tham số tốt nhất thông qua GridSearchCV kết hợp StratifiedKFold. Tập test được giữ riêng hoàn toàn cho đánh giá cuối. Cách tổ chức này vẫn tránh được rò rỉ dữ liệu vì mọi quyết định chọn tham số đều chỉ dựa trên dữ liệu train, còn test chỉ xuất hiện một lần ở bước đánh giá độc lập.

**CHƯƠNG 3. THIẾT KẾ VÀ XÂY DỰNG HỆ THỐNG**
**3.1. Tổng quan kiến trúc hệ thống**
Hệ thống được xây dựng theo kiến trúc pipeline gồm nhiều bước liên tiếp, trong đó đầu ra của bước trước là đầu vào của bước sau. Kiến trúc này giúp hệ thống rõ ràng, dễ kiểm soát và thuận lợi cho việc đánh giá từng thành phần. Hình 3.1 minh họa các bước chính trong pipeline của đề tài.

Hình 3.1. Pipeline tổng quát của hệ thống nhận dạng ảnh
Ưu điểm của cách tổ chức này là mỗi thành phần có trách nhiệm riêng. Phần tiền xử lý chỉ tập trung vào ảnh đầu vào, phần trích xuất đặc trưng chỉ quan tâm đến việc biến ảnh thành vector, phần huấn luyện chỉ làm việc với dữ liệu số đã trích xuất, còn phần ứng dụng chỉ cần dùng lại mô hình và pipeline đã có. Như thể hiện trong Hình 3.1, các bước được sắp xếp một cách tuyến tính và rõ ràng.
**3.2. Bộ dữ liệu Corel-1K**
Đề tài sử dụng bộ dữ liệu Corel-1K, một bộ dữ liệu ảnh kinh điển gồm 1000 ảnh chia đều thành 10 lớp. Mỗi lớp chứa 100 ảnh. Bộ dữ liệu này thường được sử dụng trong các bài toán phân loại ảnh và truy hồi ảnh theo nội dung do có sự đa dạng tương đối rõ về chủ đề thị giác.
Bảng 3.1 dưới đây liệt kê 10 lớp ảnh trong bộ dữ liệu Corel-1K cùng mô tả nội dung:

Bảng 3.1. Danh sách các lớp ảnh trong bộ dữ liệu Corel-1K
| **STT** | **Tên lớp** | **Nội dung** | **Số ảnh** |
| --- | --- | --- | --- |
| 1 | africans | Cảnh và con người châu Phi | 100 |
| 2 | beaches | Bãi biển | 100 |
| 3 | buildings | Công trình kiến trúc | 100 |
| 4 | buses | Xe buýt | 100 |
| 5 | dinosaurs | Khủng long | 100 |
| 6 | elephants | Voi | 100 |
| 7 | flowers | Hoa | 100 |
| 8 | food | Thực phẩm | 100 |
| 9 | horses | Ngựa | 100 |
| 10 | mountains | Núi non | 100 |

Như thể hiện trong Bảng 3.1, bộ dữ liệu có phân bố đều giữa các lớp với mỗi lớp gồm đúng 100 ảnh. Một số lớp khá đồng nhất về nội dung và màu sắc như dinosaurs hoặc flowers, trong khi một số lớp như buildings hoặc food có độ biến thiên nội bộ lớn hơn.
**3.3. Tiền xử lý dữ liệu ảnh**
Trong hệ thống hiện tại, toàn bộ ảnh được đọc bằng OpenCV và chuẩn hóa về cùng kích thước 128×128. Việc đưa tất cả ảnh về một kích thước thống nhất giúp đảm bảo rằng quá trình trích xuất đặc trưng diễn ra nhất quán, đồng thời giảm độ phức tạp trong tính toán.
Lý do chọn kích thước 128×128 là vì đây là mức đủ nhỏ để giảm chi phí xử lý nhưng vẫn đủ lớn để giữ lại cấu trúc màu cơ bản của ảnh. Quá trình tiền xử lý được cài đặt trong tệp preprocessing.py. Đoạn mã dưới đây minh họa cách đọc và resize ảnh:

| **def load_image(path, size=(128, 128)):** **    img = cv2.imread(str(path))** **    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)** **    return img** |
| --- |

**3.4. Lượng tử hóa màu**
Lượng tử hóa màu là bước chuyển từ không gian màu gốc có số mức rất lớn sang một tập màu hữu hạn nhỏ hơn. Đây là bước nền tảng vì Color Histogram và Color Correlogram đều yêu cầu ảnh đầu vào đã được mã hóa thành các chỉ số màu rời rạc.
Đối với HSV, hệ thống sử dụng 8 mức cho kênh Hue, 3 mức cho Saturation và 3 mức cho Value. Cách chia này ưu tiên chi tiết hơn cho sắc độ vì Hue thường mang thông tin phân biệt loại màu rõ nhất, trong khi Saturation và Value được gom thô hơn. Tổng cộng số màu là 8 × 3 × 3 = 72. Mỗi điểm ảnh sau đó được mã hóa thành một số nguyên trong khoảng từ 0 đến 71.
Đối với RGB, mỗi kênh được chia thành 4 mức, tạo ra tổng cộng 4³ = 64 màu. Cấu hình RGB này được dùng như một hướng so sánh với HSV nhằm khảo sát ảnh hưởng của không gian màu tới hiệu quả phân loại.
**3.5. Thiết kế đặc trưng**
***3.5.1. Histogram màu***
Đặc trưng histogram được xây dựng bằng cách đếm số lượng điểm ảnh của từng màu sau lượng tử hóa, sau đó chuẩn hóa bằng tổng số điểm ảnh. Với cấu hình HSV, vector histogram có 72 chiều; với RGB, vector histogram có 64 chiều. Đây là đặc trưng baseline, có vai trò làm mốc so sánh.
***3.5.2. Auto-Correlogram***
Đối với Auto-Correlogram, sau khi lượng tử hóa ảnh, hệ thống xét bốn khoảng cách {1, 3, 5, 7}. Ở mỗi khoảng cách, hệ thống xem xét tám hướng lân cận bao quanh một điểm ảnh và ước lượng xác suất gặp lại cùng màu ở khoảng cách đó. Kết quả của tất cả màu và tất cả khoảng cách được ghép lại thành vector đặc trưng cuối cùng.
Trong quá trình cài đặt, đề tài sử dụng phiên bản tối ưu auto_correlogram_fast, sử dụng NumPy và bincount để tăng tốc tính toán. Với cấu hình HSV 72 màu và 4 khoảng cách, vector đặc trưng có kích thước 72 × 4 = 288 chiều.
***3.5.3. Spatial Correlogram***
Spatial Correlogram là biến thể mở rộng của Auto-Correlogram. Ảnh được chia thành bốn ô theo lưới 2×2, sau đó tính correlogram riêng cho từng ô. Đồng thời, hệ thống vẫn giữ correlogram toàn ảnh. Vector cuối cùng là phép nối của năm phần: một phần toàn ảnh và bốn phần cục bộ, cho tổng số chiều là 1440.
**3.6. Tổ chức dữ liệu đặc trưng**
Sau khi trích xuất, các đặc trưng được lưu vào thư mục data/features/ dưới dạng tệp .npy. Bảng 3.2 dưới đây tổng hợp kích thước các ma trận đặc trưng đã trích xuất từ toàn bộ 1000 ảnh:

Bảng 3.2. Kích thước các ma trận đặc trưng đã trích xuất
| **Tệp .npy** | **Kích thước** | **Mô tả** |
| --- | --- | --- |
| correlogram_hsv.npy | (1000, 288) | Correlogram HSV – 72 màu × 4 khoảng cách |
| correlogram_hsv_spatial.npy | (1000, 1440) | Spatial Correlogram HSV – toàn ảnh + 2×2 lưới |
| correlogram_rgb.npy | (1000, 256) | Correlogram RGB – 64 màu × 4 khoảng cách |
| histogram_hsv.npy | (1000, 72) | Histogram HSV – 72 màu |
| histogram_rgb.npy | (1000, 64) | Histogram RGB – 64 màu |
| labels.npy | (1000,) | Nhãn lớp số của 1000 ảnh |
| class_names.npy | (10,) | Tên 10 lớp ảnh |

Như thể hiện trong Bảng 3.2, Spatial Correlogram HSV có chiều cao nhất với 1440 chiều, trong khi Histogram HSV chỉ có 72 chiều. Sự chênh lệch này phản ánh lượng thông tin bổ sung mà Correlogram cung cấp so với Histogram. Trong phiên bản hiện tại của repo, các artifact ưu tiên cho ứng dụng demo còn được namespace theo dataset profile, chẳng hạn corel-1k_correlogram_hsv_spatial.npy; các tên không tiền tố vẫn được giữ lại để tương thích với các lần chạy trước.
**3.7. Chia tập dữ liệu**
Hệ thống sử dụng một tệp metadata để lưu split cố định, nằm tại data/splits/corel-1k_split.json. Tập dữ liệu được chia theo tỷ lệ 80% cho huấn luyện và 20% cho kiểm tra. Với tổng 1000 ảnh, hệ thống có 800 ảnh train và 200 ảnh test, tương ứng 80 ảnh train và 20 ảnh test cho mỗi lớp.
Việc lưu split ra file riêng mang hai lợi ích lớn. Thứ nhất, kết quả giữa các lần chạy luôn nhất quán, giúp việc so sánh mô hình công bằng hơn. Thứ hai, ứng dụng demo có thể dùng đúng cùng một logic chia dữ liệu để chỉ truy hồi ảnh tương tự trong train split, tránh vô tình dùng ảnh của tập kiểm tra.
**3.8. Thiết kế pipeline huấn luyện**
Pipeline huấn luyện được tổ chức trong tệp train.py. Hệ thống nạp toàn bộ ma trận đặc trưng và nhãn đã lưu sẵn. Với từng thí nghiệm, hệ thống chọn đúng ma trận đặc trưng tương ứng, thực hiện GridSearchCV với StratifiedKFold trên train split để tìm tham số tốt nhất, rồi lưu mô hình cuối huấn luyện trên toàn bộ train split ra file dùng cho ứng dụng.
Cách làm này vẫn có ý nghĩa học thuật rõ ràng. Trong nhiều đồ án sinh viên, tập test đôi khi bị dùng quá sớm trong quá trình lựa chọn mô hình, dẫn đến đánh giá thiên lệch. Hệ thống hiện tại tránh được lỗi đó nhờ tách riêng vai trò của train và held-out test: tập train chịu trách nhiệm cho quá trình chọn tham số bằng cross-validation, còn test chỉ dùng cho đánh giá cuối. Các mô hình được lưu tại thư mục models/ dưới dạng .pkl, kèm theo tệp metadata .meta.json.
**3.9. Cấu trúc mã nguồn dự án**
Mã nguồn của dự án được tổ chức thành các tệp có chức năng tương đối tách bạch, như tổng hợp trong Bảng 3.3 dưới đây:

Bảng 3.3. Mô tả chức năng các tệp mã nguồn chính
| **Tệp** | **Chức năng** |
| --- | --- |
| preprocessing.py | Đọc dữ liệu ảnh, chuyển đổi không gian màu, lượng tử hóa màu |
| color_histogram.py | Cài đặt đặc trưng Color Histogram (baseline) |
| color_correlogram.py | Cài đặt Auto-Correlogram và Spatial Correlogram |
| dataset_profile.py | Khai báo dataset profile, tên artifact có namespace và helper resolve đường dẫn dữ liệu |
| feature_extraction.py | Trích xuất toàn bộ đặc trưng, lưu file .npy và tạo metadata split theo dataset profile |
| dataset_split.py | Tạo, nạp và ánh xạ metadata split train/test cố định cho pipeline |
| train.py | Huấn luyện các mô hình, tune trên train split và lưu model/meta cho ứng dụng |
| evaluate.py | Đánh giá mô hình trên tập test độc lập |
| experiment_runner.py | Hỗ trợ chạy thực nghiệm linh hoạt với nhiều chiến lược |
| evaluation_methods.py | Các phương pháp đánh giá (hold-out, k-fold, bootstrap,...) |
| app.py | Ứng dụng Streamlit demo phân loại, truy hồi ảnh tương tự và chạy lại các thí nghiệm đánh giá |

Như mô tả trong Bảng 3.3, việc phân tách rõ chức năng giúp dự án có cấu trúc tốt, dễ đọc, dễ kiểm thử và cũng thuận lợi khi viết báo cáo vì mỗi tệp tương ứng gần như với một bước trong pipeline.

**3.10. Ứng dụng minh họa bằng Streamlit**
Bên cạnh pipeline xử lý dữ liệu và huấn luyện mô hình, đề tài còn xây dựng một ứng dụng minh họa bằng Streamlit để giúp quá trình trình bày kết quả trở nên trực quan hơn. Ứng dụng này không chỉ phục vụ thao tác demo trước người xem mà còn đóng vai trò như một lớp giao diện gắn kết trực tiếp giữa mô hình đã huấn luyện, dữ liệu đặc trưng đã lưu và các báo cáo đánh giá trong thư mục results [6].
Tệp app.py hiện tổ chức giao diện thành hai tab chính. Tab thứ nhất tập trung vào bài toán nhận dạng ảnh bằng mô hình tĩnh đã huấn luyện sẵn; tab thứ hai cho phép chạy lại các cấu hình thực nghiệm với nhiều kỹ thuật lượng giá khác nhau như independent test, holdout, k-fold hay bootstrap. Nhờ đó, ứng dụng vừa có giá trị minh họa cho người dùng cuối, vừa có giá trị học thuật trong việc quan sát lại quy trình thực nghiệm.
***3.10.1. Tab nhận dạng ảnh***
Tab Nhận dạng ảnh đặt thao tác chính lên đầu giao diện: người dùng chọn không gian màu, tải ảnh cần dự đoán, quan sát cấu hình mô hình đang dùng và đọc nhanh các thông tin provenance của thí nghiệm. Trong phiên bản ổn định của dự án, cấu hình mặc định là HSV Spatial với các tham số H=8, S=3, V=3; ứng dụng hiện ưu tiên nạp mô hình corel-1k_svm_correlogram_hsv_spatial.pkl cùng vector đặc trưng corel-1k_correlogram_hsv_spatial.npy, và đây cũng là cấu hình đạt kết quả tốt nhất trong phần đánh giá offline. Đối với không gian màu RGB, ứng dụng sử dụng mô hình được huấn luyện với tệp corel-1k_svm_correlogram_rgb.pkl và các đặc trưng từ correlogram_rgb.npy để đảm bảo tương thích ngược; tuy nhiên, việc sử dụng tên gọi có phạm vi (scoped naming) được ưu tiên hơn.

Hình 3.2. Giao diện tab Nhận dạng ảnh trong ứng dụng Streamlit

Như thể hiện trong Hình 3.2, giao diện tab này cho phép điều chỉnh nhanh các tham số lượng tử hóa, đồng thời hiển thị rõ mô hình đang dùng, nguồn gốc split huấn luyện và giới hạn diễn giải khi áp dụng mô hình cho ảnh ngoài miền dữ liệu Corel-1K. Cách tổ chức này giúp phần demo bám sát đúng protocol thực nghiệm đã mô tả trong báo cáo thay vì chỉ trình diễn một kết quả rời rạc.
***3.10.2. Kết quả trực tiếp và truy hồi ảnh tương tự***
Sau khi người dùng tải ảnh lên, ứng dụng sẽ chuẩn hóa ảnh về kích thước 128x128, trích xuất vector đặc trưng Color Correlogram tương ứng, thực hiện dự đoán lớp và nếu mô hình hỗ trợ thì hiển thị luôn xác suất cho top-3 lớp gần nhất. Cùng với đó, hệ thống còn truy hồi một nhóm ảnh gần nhất trong tập train để giúp người xem hiểu được mô hình đang xếp ảnh mới gần với cụm dữ liệu nào trong không gian đặc trưng.

Hình 3.3. Màn hình kết quả trực tiếp sau khi dự đoán và truy hồi ảnh tương tự

Hình 3.3 cho thấy phần kết quả được trình bày theo ba lớp thông tin: ảnh đầu vào đã chuẩn hóa, biểu đồ vector đặc trưng và phần dự đoán/xếp hạng lớp. Bên dưới là gallery ảnh tương tự trong train split, đóng vai trò như một cơ chế đối chiếu trực quan rất hữu ích khi giải thích vì sao ảnh được gán vào lớp beaches thay vì các lớp có phân bố màu gần nhau như mountains hoặc buildings.
***3.10.3. Tab thực nghiệm thuật toán***
Tab Thực nghiệm thuật toán mở rộng ứng dụng từ vai trò demo sang vai trò công cụ kiểm tra pipeline. Tại đây, người dùng có thể chọn đặc trưng, không gian màu, mô hình và kỹ thuật lượng giá; sau đó hệ thống sẽ tự động khởi tạo lại quy trình train/evaluate tương ứng thông qua hàm run_experiment trong mã nguồn.

Hình 3.4. Giao diện tab Thực nghiệm thuật toán của ứng dụng Streamlit

Như minh họa trong Hình 3.4, ứng dụng mô tả rõ quy trình của từng kỹ thuật lượng giá, đặc biệt nhấn mạnh benchmark independent held-out test dùng file split cố định của dự án. Điều này giúp người xem phân biệt giữa mô hình tĩnh dùng để demo nhận dạng và quy trình thực nghiệm động dùng để khảo sát độ ổn định của thuật toán dưới các cách chia dữ liệu khác nhau.
***3.10.4. Kết quả thí nghiệm và artifact đã lưu***
Khi một thí nghiệm được chạy xong, ứng dụng sẽ ghi ra các artifact gồm file JSON tổng hợp metric, báo cáo văn bản và ảnh confusion matrix. Đồng thời, giao diện hiển thị lại các thống kê chính như accuracy, precision, recall, F1-score cùng thông tin về đặc trưng, không gian màu, mô hình và kỹ thuật lượng giá vừa sử dụng.

Hình 3.5. Bảng điều khiển kết quả và đường dẫn artifact sau khi chạy thí nghiệm

Hình 3.5 cho thấy tab thực nghiệm không chỉ trả về metric tổng quát mà còn hiển thị trực tiếp đường dẫn tới các artifact vừa sinh ra trong thư mục results. Chi tiết này quan trọng về mặt tái lập vì người đọc có thể lần ngược từ giao diện Streamlit về file JSON, báo cáo văn bản và ảnh confusion matrix tương ứng thay vì chỉ xem một kết quả minh họa trên màn hình.

Hình 3.6. Bảng báo cáo phân loại theo lớp

Để phân tích chất lượng theo từng lớp ảnh thay vì chỉ nhìn vào accuracy tổng quát, giao diện còn hiển thị báo cáo phân loại như trong Hình 3.6.

Từ báo cáo phân loại của cấu hình tốt nhất Spatial Correlogram HSV + SVM, có thể thấy dinosaurs và horses đạt recall 1.00, flowers đạt 0.95, trong khi beaches chỉ đạt 0.60 và mountains 0.55. Điều này cho thấy hệ thống xử lý rất tốt các lớp có màu sắc hoặc bố cục đặc trưng rõ rệt, nhưng vẫn gặp khó với các lớp cảnh tự nhiên có phân bố màu chồng lấn.

Hình 3.7. Bảng báo ma trận nhầm lẫn

Ở mức trực quan hơn, ma trận nhầm lẫn trong Hình 3.7 giúp nhận ra cụ thể ảnh của lớp nào đang bị chuyển sang lớp nào.
Đối chiếu confusion matrix với các báo cáo trong thư mục results cho thấy sai số tập trung nhiều ở nhóm beaches, buildings và mountains, tức các lớp có vùng xanh, xám, nâu xuất hiện xen kẽ với bố cục khá gần nhau. Đây cũng là nơi Spatial Correlogram phát huy lợi thế hơn Histogram vì đặc trưng này vẫn giữ lại một phần thông tin bố cục màu theo khoảng cách và theo vùng.

Nhờ màn hình kết quả như trong Hình 3.5, phần Streamlit không chỉ là giao diện trực quan mà còn trở thành một công cụ kết nối giữa mã nguồn, artifact sinh ra trong results và nội dung phân tích trong báo cáo. Đây là điểm mạnh quan trọng của dự án vì người đọc có thể kiểm tra lại toàn bộ vòng đời từ đầu vào, mô hình, phép lượng giá cho tới file kết quả cuối cùng ngay trong một giao diện thống nhất.

**CHƯƠNG 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ**
**4.1. Mục tiêu thực nghiệm**
Phần thực nghiệm của đề tài nhằm trả lời một số câu hỏi nghiên cứu chính. Thứ nhất, đặc trưng Color Correlogram có thực sự hiệu quả hơn Color Histogram trong bài toán nhận dạng ảnh trên bộ dữ liệu Corel-1K hay không. Thứ hai, không gian màu HSV và RGB khác nhau như thế nào về hiệu quả. Thứ ba, trong số các mô hình học máy đã chọn, mô hình nào phù hợp nhất. Thứ tư, việc mở rộng từ Correlogram toàn ảnh sang Spatial Correlogram có mang lại cải thiện rõ rệt hay không.
**4.2. Môi trường thực nghiệm**
Hệ thống được triển khai bằng Python, sử dụng các thư viện chính gồm NumPy, OpenCV, Pillow, scikit-learn, matplotlib, seaborn, joblib, tqdm và Streamlit. Dự án được tổ chức theo dạng repo với cấu trúc thư mục rõ ràng gồm dữ liệu, đặc trưng, mô hình, mã nguồn, kết quả và notebook minh họa. Việc sử dụng tệp .npy cho đặc trưng và .pkl cho mô hình giúp quá trình thực nghiệm nhanh hơn đáng kể so với việc xử lý lại từ đầu ở mỗi lần chạy [5][6][7][8].
**4.3. Quy trình thực nghiệm**
Quy trình thực nghiệm trong đề tài diễn ra theo các bước sau. Đầu tiên, toàn bộ ảnh của Corel-1K được đọc và resize về kích thước 128×128. Sau đó, ảnh được lượng tử hóa trong không gian HSV hoặc RGB. Tiếp theo, hệ thống trích xuất các loại đặc trưng đã định nghĩa.
Sau khi có đặc trưng, dữ liệu được chia theo metadata cố định với tỷ lệ 80/20 cho train/test. Mỗi mô hình được huấn luyện trên tập train bằng GridSearchCV kết hợp StratifiedKFold để tìm tham số tốt nhất. Sau đó, mô hình với tham số đã chọn được huấn luyện trên toàn bộ train split và kiểm tra trên tập test độc lập nhằm phản ánh khả năng khái quát hóa của hệ thống.

**4.4. Các cấu hình tham số được tìm kiếm**
Bảng 4.1 dưới đây tổng hợp không gian tìm kiếm tham số của từng mô hình trong GridSearchCV:

Bảng 4.1. Không gian tham số tìm kiếm GridSearchCV cho từng mô hình
| **Mô hình** | **Tham số** | **Giá trị tìm kiếm** |
| --- | --- | --- |
| SVM | C | {0.1, 1, 10, 100} |
| SVM | gamma | {scale, 0.01, 0.001} |
| SVM | kernel | {rbf, linear} |
| KNN | n_neighbors | {3, 5, 7, 9, 11} |
| KNN | weights | {uniform, distance} |
| KNN | metric | {euclidean, manhattan} |

Như tổng hợp trong Bảng 4.1, không gian tìm kiếm tham số khá rộng đối với SVM và KNN, đảm bảo tìm được cấu hình phù hợp nhất với từng loại đặc trưng.
**4.5. Kết quả huấn luyện và cross-validation trên train split**
Kết quả lưu trong results/corel-1k_training_results.json cho thấy sự khác biệt rõ giữa các cấu hình. Vì pipeline hiện tại không duy trì validation split riêng, Bảng 4.2 tổng hợp train CV accuracy trên train split cùng tham số tốt nhất của từng mô hình.

Bảng 4.2. Kết quả huấn luyện - Train CV Accuracy và tham số tốt nhất
| Phương pháp | Train CV Acc. | Tham số tốt nhất | Split train cuối |
| --- | --- | --- | --- |
| Spatial Correlogram HSV + SVM | 0.8763 | linear, C=0.1 | train (800) |
| Correlogram HSV + SVM | 0.8500 | rbf, C=100, g=0.001 | train (800) |
| Correlogram HSV + KNN | 0.7800 | k=5, Manhattan | train (800) |
| Correlogram RGB + SVM | 0.8250 | C=10, gamma=0.001, kernel=rbf | train (800) |
| Histogram HSV + SVM | 0.7825 | rbf, C=100, g=0.01 | train (800) |

Bảng 4.2 cho thấy Spatial Correlogram HSV kết hợp SVM đạt train CV accuracy = 0.8763, cao nhất trong các cấu hình. Correlogram HSV + SVM đạt 0.8500, Correlogram RGB + SVM đạt 0.8250, trong khi Histogram HSV + SVM và Correlogram HSV + KNN chỉ quanh mức 0.78. Điều này cho thấy việc bổ sung thông tin không gian của màu vẫn tạo lợi thế rõ ràng ngay từ giai đoạn cross-validation trên tập train.
Cũng đáng chú ý là tham số tốt nhất tìm được cho Spatial Correlogram HSV + SVM là C=0.1, gamma=scale, kernel=linear, trong khi Correlogram HSV + SVM dùng C=100, gamma=0.001, kernel=rbf. Correlogram RGB + SVM dùng C=10, gamma=0.001, kernel=rbf. Sự khác biệt này gợi ý rằng Spatial Correlogram với 1440 chiều đã tạo ra một không gian đặc trưng có mức phân tách tuyến tính tốt hơn.
**4.6. Kết quả đánh giá trên tập kiểm tra độc lập**
Kết quả chính thức trên tập test được lưu trong results/corel-1k_evaluation_summary.json. Bảng 4.3 dưới đây tổng hợp đầy đủ các chỉ số đánh giá:

Bảng 4.3. Kết quả đánh giá trên tập kiểm tra độc lập (200 ảnh test)
| Phương pháp | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- |
| Spatial Correlogram HSV + SVM | 84.50% | 0.8462 | 0.8450 | 0.8442 |
| Correlogram HSV + SVM | 82.50% | 0.8313 | 0.8250 | 0.8259 |
| Correlogram HSV + KNN | 78.00% | 0.8233 | 0.7800 | 0.7790 |
| Correlogram RGB + SVM | 82.00% | 0.8294 | 0.8200 | 0.8188 |
| Histogram HSV + SVM | 78.00% | 0.7798 | 0.7800 | 0.7778 |

Số liệu trong Bảng 4.3 được trích trực tiếp từ tệp corel-1k_evaluation_summary.json trong dự án. Hai cấu hình tốt nhất trên tập test là Spatial Correlogram HSV + SVM với accuracy 84.50% và Correlogram HSV + SVM với 82.50%. Histogram HSV + SVM và Correlogram HSV + KNN cùng đạt 78.00%, thấp hơn rõ rệt so với các biến thể Correlogram mạnh hơn.
Nếu đọc sâu hơn báo cáo phân loại theo từng lớp của cấu hình Spatial Correlogram HSV + SVM trong thư mục results, có thể thấy mô hình mạnh nhất hiện tại đạt kết quả gần như tuyệt đối ở dinosaurs, horses và flowers, nhưng còn yếu hơn ở beaches, buildings và mountains. Mẫu hình này phù hợp với bản chất của bộ dữ liệu: các lớp đối tượng nổi bật thường có tín hiệu màu và bố cục khá đặc trưng, trong khi các lớp cảnh thiên nhiên dễ chia sẻ những vùng màu xanh, nâu, xám và có biên quyết định kém sắc nét hơn.
Một đối chiếu đáng chú ý là Correlogram RGB + SVM nâng recall của mountains lên 0.80 nhưng đồng thời kéo recall của buses xuống 0.60, trong khi Spatial Correlogram HSV + SVM vẫn giữ buses ở 0.90 và cho accuracy tổng thể cao hơn. Điều này gợi ý HSV vẫn là không gian màu cân bằng hơn cho repo hiện tại, còn việc bổ sung spatial correlogram đem lại cải thiện ổn định hơn so với chỉ chuyển sang RGB.
**4.7. Phân tích kết quả**
***4.7.1. So sánh giữa Histogram và Correlogram***
Kết quả thực nghiệm cho thấy Correlogram vượt trội hơn Histogram trên cùng một không gian màu HSV. Cụ thể, Histogram HSV + SVM đạt accuracy 78.00%, trong khi Correlogram HSV + SVM đạt 82.50% và Spatial Correlogram HSV + SVM đạt 84.50%. Chênh lệch khoảng 4.5 đến 6.5 điểm phần trăm này đủ lớn để khẳng định rằng thông tin tương quan không gian của màu mang lại lợi ích thực sự.
Về mặt lý thuyết, điều này hoàn toàn hợp lý. Histogram chỉ phản ánh "ảnh có những màu gì" và "mỗi màu xuất hiện nhiều hay ít", nhưng không cho biết các màu đó nằm ở đâu và tương tác với nhau thế nào trong bố cục ảnh. Correlogram bổ sung thêm chiều thông tin này, vì vậy có khả năng phân biệt tốt hơn những lớp ảnh có histogram gần giống nhau nhưng có cấu trúc không gian màu khác nhau.
***4.7.2. So sánh giữa không gian màu HSV và RGB***
Khi so sánh Correlogram HSV + SVM với Correlogram RGB + SVM, kết quả cho thấy HSV nhỉnh hơn nhẹ: 82.50% so với 82.00% accuracy trên tập test. Chênh lệch không lớn, nhưng cấu hình HSV vẫn giữ ưu thế chung, đặc biệt khi mở rộng sang Spatial Correlogram thì đây cũng là hướng đạt kết quả tốt nhất toàn bộ thí nghiệm.
***4.7.3. So sánh giữa các mô hình học máy***
SVM cho thấy hiệu quả ổn định và cao ở hầu hết các cấu hình. Đây là kết quả hợp lý vì SVM thường phù hợp với dữ liệu có số chiều tương đối cao và số mẫu vừa phải. Spatial Correlogram có tới 1440 chiều, và SVM vẫn xử lý tốt đặc trưng này.
KNN cho kết quả thấp hơn các cấu hình mạnh nhất với accuracy 78.00%. Các khoảng cách trong không gian nhiều chiều thường trở nên kém phân biệt hơn, khiến việc bỏ phiếu dựa trên láng giềng gần không còn hiệu quả như trong không gian thấp chiều.
Kết quả thực nghiệm cho thấy SVM là lựa chọn hiệu quả nhất khi đi cùng đặc trưng Color Correlogram, đặc biệt ở cấu hình spatial HSV đạt 84.50% accuracy trên tập test. Trong khi đó, KNN vẫn cung cấp một mốc so sánh hữu ích nhưng giảm hiệu quả khi số chiều đặc trưng tăng lên.
***4.7.4. Vai trò của Spatial Correlogram***
Spatial Correlogram là cải tiến đáng chú ý nhất trong hệ thống hiện tại. Việc kết hợp đặc trưng toàn ảnh với các vùng con 2×2 giúp mô hình giữ được thêm thông tin về bố cục màu. Kết quả train CV cho Spatial Correlogram là 0.8763, cao nhất trong tất cả các cấu hình, và trên tập test nó cũng đứng đầu với accuracy 84.50%.
**4.8. Phân tích chênh lệch giữa Train CV và Test**
Một điểm đáng chú ý là train CV accuracy thường cao hơn held-out test accuracy. Ví dụ, Spatial Correlogram HSV + SVM đạt 87.63% ở train CV nhưng 84.50% trên test, chênh lệch 3.13 điểm phần trăm. Correlogram HSV + KNN giữ nguyên ở mức 78.00%, còn Histogram HSV + SVM chỉ giảm 0.25 điểm. Điều này là bình thường trong thực nghiệm học máy: cross-validation trên train split có xu hướng lạc quan hơn một chút so với đánh giá cuối trên dữ liệu chưa từng được dùng để chọn tham số.
Quan trọng hơn, tất cả mô hình đều được đánh giá trên cùng một test split độc lập gồm 200 ảnh, nên sự so sánh giữa chúng vẫn công bằng. Sự sụt giảm nhìn chung không lớn và kết quả test vẫn đủ thuyết phục để kết luận về hiệu quả tương đối giữa các phương pháp.

**4.9. Tổng hợp kết quả và nhận xét chung**
Bảng 4.4 dưới đây tổng hợp so sánh trực tiếp giữa kết quả train CV và test cho tất cả các cấu hình:
Bảng 4.4. So sánh Train CV Accuracy và Test Accuracy giữa các cấu hình
| Phương pháp | Train CV Accuracy | Test Accuracy | Chênh lệch |
| --- | --- | --- | --- |
| Spatial Corr. HSV + SVM | 87.63% | 84.50% | -3.13% |
| Correlogram HSV + SVM | 85.00% | 82.50% | -2.50% |
| Correlogram HSV + KNN | 78.00% | 78.00% | 0.00% |
| Correlogram RGB + SVM | 82.50% | 82.00% | -0.50% |
| Histogram HSV + SVM | 78.25% | 78.00% | -0.25% |

Nhìn vào Bảng 4.4, Correlogram HSV + KNN ổn định nhất với chênh lệch 0.00%, tiếp theo là Histogram HSV + SVM (-0.25%) và Correlogram RGB + SVM (-0.50%). Tuy nhiên, nếu xét đồng thời cả độ chính xác lẫn độ ổn định, Spatial Correlogram HSV + SVM vẫn là cấu hình thuyết phục nhất trong phiên bản chính thức của dự án.
**4.10. Hạn chế của thực nghiệm**
Dù hệ thống cho kết quả tương đối tốt, vẫn còn một số hạn chế cần nhìn nhận. Trước hết, bộ dữ liệu Corel-1K chỉ có 1000 ảnh và 10 lớp, quy mô còn khá nhỏ so với các bài toán nhận dạng ảnh hiện đại. Thứ hai, đặc trưng chủ yếu dựa trên màu sắc nên hệ thống sẽ gặp khó khăn khi hai lớp có phân bố màu gần nhau nhưng khác nhau chủ yếu ở hình dạng hoặc kết cấu. Thứ ba, số lượng cấu hình tham số được thử tuy đã đủ để so sánh nhưng vẫn chưa quá rộng. Cuối cùng, ứng dụng demo buộc phải gán ảnh vào một trong 10 lớp gần nhất dù ảnh đó không thực sự thuộc bất kỳ lớp nào.

**CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN**
**5.1. Kết luận**
Đề tài "Nghiên cứu về sử dụng phương pháp Tương quan màu sắc (Color Correlograms) cho bài toán nhận dạng ảnh bằng phương pháp học máy" đã xây dựng và đánh giá một hệ thống nhận dạng ảnh dựa trên đặc trưng màu, trong đó trọng tâm là Color Correlogram. Hệ thống được triển khai đầy đủ từ khâu đọc dữ liệu, tiền xử lý, lượng tử hóa màu, trích xuất đặc trưng, chia tập dữ liệu, huấn luyện mô hình, đánh giá kết quả cho đến xây dựng ứng dụng minh họa.
Về mặt lý thuyết, báo cáo đã làm rõ được sự khác biệt giữa Color Histogram và Color Correlogram. Trong khi Histogram chỉ mô tả tần suất xuất hiện của màu, Correlogram bổ sung thêm quan hệ không gian giữa các màu, từ đó biểu diễn ảnh giàu thông tin hơn. Việc mở rộng sang Spatial Correlogram còn cho phép hệ thống nắm bắt bố cục tương đối của màu theo từng vùng ảnh.
Về mặt thực nghiệm, kết quả trên bộ dữ liệu Corel-1K cho thấy các đặc trưng Correlogram đều cho hiệu quả tốt hơn đáng kể so với Histogram. Cấu hình tốt nhất là Spatial Correlogram HSV kết hợp SVM, đạt accuracy 84.50% trên tập kiểm tra độc lập gồm 200 ảnh. Histogram HSV + SVM chỉ đạt 78.00%, thấp hơn 6.5 điểm phần trăm, đây là bằng chứng rõ ràng cho thấy việc sử dụng tương quan màu sắc là một hướng phù hợp.
Ngoài ra, đề tài cũng chỉ ra rằng SVM là mô hình hoạt động ổn định và hiệu quả với đặc trưng Color Correlogram. KNN cho kết quả thấp hơn do hạn chế trong không gian đặc trưng nhiều chiều, nhưng vẫn đóng vai trò là mốc đối sánh cần thiết để làm rõ ưu thế của SVM.
Từ góc độ học tập và nghiên cứu, đề tài đạt được giá trị ở chỗ không chỉ xây dựng một hệ thống chạy được, mà còn phân tích được mối liên hệ giữa cách biểu diễn đặc trưng và hiệu quả phân loại. Đây là một bài học quan trọng trong khai phá dữ liệu đa phương tiện và thị giác máy tính.

**5.2. Hướng phát triển**
Mặc dù đã đạt được các mục tiêu cơ bản, hệ thống vẫn còn nhiều hướng phát triển tiềm năng:
Mở rộng tập dữ liệu sang quy mô lớn hơn và đa dạng hơn để đánh giá độ khái quát của Color Correlogram trong các điều kiện thực tế. Khi dữ liệu lớn hơn, có thể kiểm tra xem đặc trưng này còn giữ được ưu thế hay không.
Kết hợp Color Correlogram với các nhóm đặc trưng khác như đặc trưng kết cấu (texture) hoặc hình dạng (shape). Trong nhiều trường hợp, màu sắc thôi chưa đủ để phân biệt lớp ảnh, đặc biệt khi các lớp có phân bố màu gần nhau.
Thử nghiệm thêm các chiến lược giảm chiều (PCA, LDA) hoặc lựa chọn đặc trưng nhằm làm gọn vector Spatial Correlogram 1440 chiều nhưng vẫn giữ hiệu quả phân loại.
Mở rộng phần báo cáo thực nghiệm bằng cách chạy có hệ thống repeated hold-out, k-fold cross-validation hoặc bootstrap trên nhiều cấu hình, rồi đối chiếu độ ổn định giữa các protocol này. Hệ thống hiện đã hỗ trợ sẵn các phương pháp đó ở mức mã nguồn và giao diện thực nghiệm.
Kết hợp phương pháp đặc trưng thủ công với các mô hình hiện đại. Color Correlogram có thể được sử dụng như một nhánh đặc trưng bổ sung bên cạnh đặc trưng học sâu, hoặc dùng trong các hệ thống lai để tăng tính giải thích.
Phát triển phần demo thành một hệ thống truy hồi ảnh hoàn chỉnh hơn, cho phép người dùng tìm kiếm theo nội dung và không chỉ dừng ở dự đoán lớp.

**TÀI LIỆU THAM KHẢO**
[1] J. Huang, S. R. Kumar, M. Mitra, W. J. Zhu, and R. Zabih, "Image Indexing Using Color Correlograms," Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 1997.
[2] C. Cortes and V. Vapnik, "Support-Vector Networks," Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.
[3] T. Cover and P. Hart, "Nearest Neighbor Pattern Classification," IEEE Transactions on Information Theory, vol. 13, no. 1, pp. 21–27, 1967.
[4] R. C. Gonzalez and R. E. Woods, Digital Image Processing. Pearson, nhiều lần tái bản.
[5] scikit-learn: Machine Learning in Python. GridSearchCV, StratifiedKFold, and metrics documentation. https://scikit-learn.org/
[6] Streamlit: The fastest way to build and share data apps. https://streamlit.io/
[7] OpenCV: Open Source Computer Vision Library. https://opencv.org/
[8] SciPy: Scientific Computing Tools for Python. Distance metrics and cdist documentation. https://scipy.org/
