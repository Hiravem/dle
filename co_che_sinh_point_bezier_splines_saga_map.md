
markdown_content = """# Hướng Dẫn Triển Khai Cơ Chế Sinh Point Tự Động Theo Đường Cong (Bezier Curve / Splines) Cho Saga Map

## 1. Giới Thiệu Tổng Quan

Trong các trò chơi thể loại **Saga Map** (*Candy Crush*, *Homescapes*, *Toon Blast*), bản đồ hiển thị dạng các nút màn chơi (Level Nodes) nối tiếp nhau trên một con đường uốn lượn. 

Thay vì kéo thả thủ công từng nút bấm trên từng khung hình, việc **Sinh tự động các Point theo đường cong (Bezier Curve / Spline)** giúp:
1. **Tiết kiệm thời gian thiết kế:** Chỉ cần vẽ một đường cong uốn lượn, hệ thống tự động rải $N$ điểm màn chơi lên đường cong đó.
2. **Linh hoạt chỉnh sửa:** Thay đổi hình dạng đường cong thì vị trí toàn bộ các nút tự động cập nhật theo.
3. **Mở rộng vô hạn:** Kết hợp với cơ chế **Chunking** để ghép nối liên tục các đoạn đường cong mới phía trên.

---

## 2. Lý Thuyết Toán Học: Đường Cong Bezier Bậc 3 (Cubic Bezier Curve)

Đường cong Bezier bậc 3 được cấu tạo từ **4 điểm kiểm soát (Control Points)**:
* **$P_0$**: Điểm bắt đầu (Start Point)
* $P_1$: Điểm điều khiển 1 (Control Point 1 - kéo hướng đường cong từ $P_0$)
* $P_2$: Điểm điều khiển 2 (Control Point 2 - kéo hướng đường cong về $P_3$)
* **$P_3$**: Điểm kết thúc (End Point)

### Phương trình toán học
Với tham số $t \in [0, 1]$ (biểu thị tỷ lệ % quãng đường đi từ $P_0$ đến $P_3$):

$$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3$$

---

## 3. Triển Khai Thủ Công Bằng C# (Unity)

### Script 1: `BezierCurve.cs`
Lớp tiện ích thực hiện công thức toán học Bezier.


```

```text
File created successfully: co_che_sinh_point_bezier_splines_saga_map.md

```csharp
using UnityEngine;

public static class BezierCurve
{
    /// <summary>
    /// Tính toán vị trí Vector3 trên đường cong Bezier bậc 3 tại thời điểm t [0, 1]
    /// </summary>
    public static Vector3 GetCubicPoint(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        t = Mathf.Clamp01(t);
        float oneMinusT = 1f - t;

        return Mathf.Pow(oneMinusT, 3) * p0 +
               3f * Mathf.Pow(oneMinusT, 2) * t * p1 +
               3f * oneMinusT * Mathf.Pow(t, 2) * p2 +
               Mathf.Pow(t, 3) * p3;
    }

    /// <summary>
    /// Tính hướng vector tiếp tuyến (Tangent) tại thời điểm t
    /// </summary>
    public static Vector3 GetCubicFirstDerivative(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        t = Mathf.Clamp01(t);
        float oneMinusT = 1f - t;

        return 3f * Mathf.Pow(oneMinusT, 2) * (p1 - p0) +
               6f * oneMinusT * t * (p2 - p1) +
               3f * Mathf.Pow(t, 2) * (p3 - p2);
    }
}

```

---

### Script 2: `BezierNodeSpawner.cs`

Gắn vào từng **Map Chunk Prefab** để tự động rải các nút màn chơi.

```csharp
using System.Collections.Generic;
using UnityEngine;

public class BezierNodeSpawner : MonoBehaviour
{
    [Header("4 Điểm Khống Chế Đường Cong")]
    public Transform p0; // Điểm bắt đầu
    public Transform p1; // Tay cầm điều khiển 1
    public Transform p2; // Tay cầm điều khiển 2
    public Transform p3; // Điểm kết thúc

    [Header("Cài Đặt Nút Màn Chơi")]
    public GameObject levelNodePrefab; // Prefab UI Button hoặc World Object
    public Transform nodeContainer;     // Trục chứa các node vừa tạo
    public int numberOfNodes = 6;       // Số lượng point muốn đặt trên đường cong

    [HideInInspector]
    public List<GameObject> spawnedNodes = new List<GameObject>();

    /// <summary>
    /// Hàm khởi tạo các Point trên đường cong và đánh số Level
    /// </summary>
    public int InitializeNodes(int startingLevelNumber)
    {
        // Xóa các node cũ nếu có
        ClearNodes();

        if (numberOfNodes <= 1) return startingLevelNumber;

        int currentLevel = startingLevelNumber;

        for (int i = 0; i < numberOfNodes; i++)
        {
            // Tính toán tham số t chia đều từ 0.0 đến 1.0
            float t = (float)i / (numberOfNodes - 1);

            // Lấy tọa độ point từ đường cong
            Vector3 spawnPosition = BezierCurve.GetCubicPoint(
                p0.position, p1.position, p2.position, p3.position, t
            );

            // Sinh ra nút
            Transform parent = nodeContainer != null ? nodeContainer : transform;
            GameObject node = Instantiate(levelNodePrefab, spawnPosition, Quaternion.identity, parent);
            node.name = $"LevelNode_{currentLevel}";

            // Gán dữ liệu cho Nút (Ví dụ: Số hiển thị trên nút)
            LevelNodeUI nodeUI = node.GetComponent<LevelNodeUI>();
            if (nodeUI != null)
            {
                nodeUI.SetLevelData(currentLevel);
            }

            spawnedNodes.Add(node);
            currentLevel++;
        }

        return currentLevel; // Trả về số level tiếp theo cho Chunk sau
    }

    private void ClearNodes()
    {
        foreach (var node in spawnedNodes)
        {
            if (node != null) Destroy(node);
        }
        spawnedNodes.Clear();
    }

    // Vẽ đường cong xem trước trong cửa sổ Scene
    private void OnDrawGizmos()
    {
        if (p0 == null || p1 == null || p2 == null || p3 == null) return;

        Gizmos.color = Color.cyan;
        Vector3 previousPoint = p0.position;
        int resolution = 30;

        for (int i = 1; i <= resolution; i++)
        {
            float t = (float)i / resolution;
            Vector3 currentPoint = BezierCurve.GetCubicPoint(p0.position, p1.position, p2.position, p3.position, t);
            Gizmos.DrawLine(previousPoint, currentPoint);
            previousPoint = currentPoint;
        }

        // Vẽ dây nối điểm điều khiển
        Gizmos.color = Color.yellow;
        Gizmos.DrawLine(p0.position, p1.position);
        Gizmos.DrawLine(p2.position, p3.position);
    }
}

```

---

## 4. Giải Quyết Vấn Đề Khoảng Cách Không Đều (Arc-Length Parameterization)

### Vấn đề

Khi tham số $t$ tăng đều (0.0, 0.2, 0.4...), độ dài thực tế của đoạn đường cong tương ứng **không bằng nhau**. Nếu bạn kéo hai điểm điều khiển $P_1, P_2$ ra xa, các nút ở giữa sẽ bị dãn cách quá rộng, trong khi hai đầu bị cụm sát lại.

### Giải pháp: Khoảng cách theo độ dài thực (Arc-Length Equal Spacing)

Chúng ta ước lượng chiều dài đường cong bằng cách chia nhỏ thành $M$ đoạn thẳng nhỏ (đoạn xấp xỉ), sau đó rải point dựa trên **khoảng cách bằng nhau thực tế**.

```csharp
public List<Vector3> GetEquidistantPoints(int count)
{
    List<Vector3> points = new List<Vector3>();
    if (count <= 1) return points;

    // Bước 1: Chia nhỏ đường cong để tính tổng độ dài thực tế
    int steps = 100;
    float totalLength = 0f;
    Vector3 prevPoint = p0.position;
    float[] accumulatedLengths = new float[steps + 1];
    accumulatedLengths[0] = 0f;

    for (int i = 1; i <= steps; i++)
    {
        float t = (float)i / steps;
        Vector3 currPoint = BezierCurve.GetCubicPoint(p0.position, p1.position, p2.position, p3.position, t);
        totalLength += Vector3.Distance(prevPoint, currPoint);
        accumulatedLengths[i] = totalLength;
        prevPoint = currPoint;
    }

    // Bước 2: Chia đều tổng độ dài cho (count - 1) khoảng
    float segmentLength = totalLength / (count - 1);

    for (int i = 0; i < count; i++)
    {
        float targetDistance = i * segmentLength;
        
        // Tìm khoảng t tương ứng với targetDistance bằng Tra cứu (Look-up)
        float t = FindTForDistance(targetDistance, accumulatedLengths, steps);
        Vector3 point = BezierCurve.GetCubicPoint(p0.position, p1.position, p2.position, p3.position, t);
        points.Add(point);
    }

    return points;
}

private float FindTForDistance(float targetDist, float[] lengths, int steps)
{
    for (int i = 0; i < steps; i++)
    {
        if (targetDist >= lengths[i] && targetDist <= lengths[i + 1])
        {
            // Nội suy tuyến tính trong phân đoạn nhỏ
            float segmentFraction = (targetDist - lengths[i]) / (lengths[i + 1] - lengths[i]);
            return (i + segmentFraction) / steps;
        }
    }
    return 1f;
}

```

---

## 5. Phương Pháp Hiện Đại: Dùng Package `Unity Splines`

Nếu dự án của bạn sử dụng Unity phiên bản mới (2021.3 trở lên), Unity hỗ trợ sẵn Package chính thức **Splines** (`com.unity.splines`).

### Ưu điểm

* Vẽ và điều chỉnh Spline trực quan ngay trong cửa sổ Scene View.
* Hỗ trợ tự động tính toán khoảng cách đều (**Uniform Spacing / Arc Length Evaluation**).
* Tối ưu hiệu năng cao.

### Script rải point bằng Unity Splines API

```csharp
using UnityEngine;
using UnityEngine.Splines;

public class SplineNodeSpawner : MonoBehaviour
{
    public SplineContainer splineContainer; // Component SplineContainer của Unity
    public GameObject levelNodePrefab;
    public int numberOfNodes = 8;

    public int InitializeNodesWithSpline(int startLevel)
    {
        if (splineContainer == null) return startLevel;

        Spline spline = splineContainer.Spline;
        float splineLength = splineContainer.CalculateLength();

        int currentLevel = startLevel;

        for (int i = 0; i < numberOfNodes; i++)
        {
            // Tỷ lệ khoảng cách cách đều
            float distance = (float)i / (numberOfNodes - 1) * splineLength;
            
            // Tính tham số t tương ứng với độ dài thực tế
            float t = distance / splineLength;

            // Lấy vị trí trên Spline theo không gian World
            Vector3 position = splineContainer.EvaluatePosition(t);

            // Sinh nút bấm
            GameObject node = Instantiate(levelNodePrefab, position, Quaternion.identity, transform);
            
            LevelNodeUI nodeUI = node.GetComponent<LevelNodeUI>();
            if (nodeUI != null)
            {
                nodeUI.SetLevelData(currentLevel);
            }

            currentLevel++;
        }

        return currentLevel;
    }
}

```

---

## 6. Vẽ Đường Chỉ Nối Giữa Các Point (Dotted Line Connector)

Để tạo hiệu ứng đường nét đứt nối các nút level giống *Candy Crush*:

1. Sử dụng Component **`LineRenderer`** trong Unity.
2. Gán Material có Texture đường chấm đứt và để chế độ **Texture Mode = Tile**.
3. Cập nhật danh sách điểm `LineRenderer.SetPositions()` bằng danh sách vị trí các Level Node đã sinh ra.

```csharp
public void DrawPathLine(List<Vector3> nodePositions)
{
    LineRenderer lineRenderer = GetComponent<LineRenderer>();
    if (lineRenderer == null || nodePositions == null || nodePositions.Count == 0) return;

    lineRenderer.positionCount = nodePositions.Count;
    lineRenderer.SetPositions(nodePositions.ToArray());
}

```

---

## 7. Quy Trình Ghép Nối Các Chunk Vô Hạn (Saga Map Architecture)

```
[ Chunk 1: Point 1 - 10 ]  ---> EndPoint = StartPoint của Chunk 2
           |
[ Chunk 2: Point 11 - 20 ] ---> EndPoint = StartPoint của Chunk 3
           |
[ Chunk 3: Point 21 - 30 ] ---> ...

```

1. **StartPoint & EndPoint:**
* Đảm bảo $P_0$ (StartPoint) của Chunk mới phải khớp hoàn toàn với vị trí $P_3$ (EndPoint) của Chunk liền trước.


2. **Quản lý biến đếm Level:**
* Biến `globalCurrentLevel` được lưu ở `SagaMapManager`. Khi sinh ra Chunk mới, truyền `globalCurrentLevel` vào hàm `InitializeNodes()`, nhận lại con số mới để chuẩn bị cho Chunk tiếp theo.


3. **Dynamic Loading:**
* Giữ lại các Chunk đã mở. Chỉ vô hiệu hóa (`SetActive(false)`) các Chunk nằm quá xa tầm mắt Camera để tiết kiệm tài nguyên Rendering.



---

## 8. Tóm Tắt Checklist Triển Khai

* [x] Tạo Prefab Nút màn chơi (LevelNodeUI) hỗ trợ nhận tham số Số Level, Trạng thái (Khóa / Mở / Đã hoàn thành).
* [x] Lựa chọn giải pháp: Tự viết toán Bezier (cho tùy biến cao) hoặc Dùng Unity Splines Package (cho thao tác trực quan).
* [x] Xử lý Arc-Length Parameterization để khoảng cách giữa các nút luôn bằng nhau bất kể độ cong.
* [x] Thiết lập khớp tọa độ điểm đầu/cuối ($P_0$ và $P_3$) giữa các Chunk kế tiếp.
* [x] Gắn `LineRenderer` vẽ đường dẫn liên kết các nút.
"""


