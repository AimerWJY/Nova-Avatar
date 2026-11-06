import Foundation
import Vision
import ImageIO

struct Point: Codable {
    var x: Double
    var y: Double
    var confidence: Float
}

struct Frame: Codable {
    var file: String
    var body: [String: Point]
    var outerLips: [Point]
    var innerLips: [Point]
    var hands: [String: [String: Point]]?
}

func handPoints(_ image: CGImage, wrist: Point) throws -> [String: Point] {
    let width = Double(image.width), height = Double(image.height)
    let rect = CGRect(x: (wrist.x - 0.17) * width, y: (wrist.y - 0.12) * height,
                      width: 0.34 * width, height: 0.31 * height).integral
        .intersection(CGRect(x: 0, y: 0, width: width, height: height))
    guard let crop = image.cropping(to: rect) else { return [:] }
    let request = VNDetectHumanHandPoseRequest()
    request.maximumHandCount = 1
    try VNImageRequestHandler(cgImage: crop).perform([request])
    guard let detected = request.results?.first else { return [:] }
    let recognized = try detected.recognizedPoints(.all)
    let joints: [(String, VNHumanHandPoseObservation.JointName)] = [
        ("wrist", .wrist),
        ("thumb_cmc", .thumbCMC), ("thumb_mp", .thumbMP), ("thumb_ip", .thumbIP), ("thumb_tip", .thumbTip),
        ("index_mcp", .indexMCP), ("index_pip", .indexPIP), ("index_dip", .indexDIP), ("index_tip", .indexTip),
        ("middle_mcp", .middleMCP), ("middle_pip", .middlePIP), ("middle_dip", .middleDIP), ("middle_tip", .middleTip),
        ("ring_mcp", .ringMCP), ("ring_pip", .ringPIP), ("ring_dip", .ringDIP), ("ring_tip", .ringTip),
        ("little_mcp", .littleMCP), ("little_pip", .littlePIP), ("little_dip", .littleDIP), ("little_tip", .littleTip),
    ]
    var points: [String: Point] = [:]
    for (name, joint) in joints {
        guard let point = recognized[joint] else { continue }
        points[name] = Point(x: (rect.minX + point.x * rect.width) / width,
                             y: (rect.minY + (1 - point.y) * rect.height) / height,
                             confidence: point.confidence)
    }
    return points
}

func extract(_ url: URL, includeHands: Bool) throws -> Frame {
    let body = VNDetectHumanBodyPoseRequest()
    let face = VNDetectFaceLandmarksRequest()
    body.usesCPUOnly = true
    face.usesCPUOnly = true
    try VNImageRequestHandler(url: url).perform([body, face])
    var frame = Frame(file: url.lastPathComponent, body: [:], outerLips: [], innerLips: [])
    if let person = body.results?.first {
        let points = try person.recognizedPoints(.all)
        let joints: [(String, VNHumanBodyPoseObservation.JointName)] = [
            ("left_shoulder", .leftShoulder), ("right_shoulder", .rightShoulder),
            ("left_elbow", .leftElbow), ("right_elbow", .rightElbow),
            ("left_wrist", .leftWrist), ("right_wrist", .rightWrist),
        ]
        for (name, key) in joints {
            if let point = points[key], point.confidence > 0 {
                frame.body[name] = Point(x: point.x, y: 1 - point.y, confidence: point.confidence)
            }
        }
    }
    if let detected = face.results?.first, let landmarks = detected.landmarks {
        let box = detected.boundingBox
        func global(_ region: VNFaceLandmarkRegion2D?) -> [Point] {
            (region?.normalizedPoints ?? []).map { point in
                Point(x: box.minX + point.x * box.width,
                      y: 1 - (box.minY + point.y * box.height), confidence: detected.confidence)
            }
        }
        frame.outerLips = global(landmarks.outerLips)
        frame.innerLips = global(landmarks.innerLips)
    }
    if includeHands {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NSError(domain: "Cannot decode PNG: \(url.lastPathComponent)", code: 3)
        }
        var hands: [String: [String: Point]] = [:]
        for side in ["left", "right"] {
            if let wrist = frame.body[side + "_wrist"], wrist.confidence >= 0.25 {
                hands[side] = try handPoints(image, wrist: wrist)
            }
        }
        frame.hands = hands
    }
    return frame
}

do {
    let args = CommandLine.arguments
    guard args.count == 3 || (args.count == 4 && args[3] == "--hands") else {
        throw NSError(domain: "Usage: vision_avatar_tracks INPUT_PNG_DIRECTORY OUTPUT_JSON [--hands]", code: 1)
    }
    let directory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension.lowercased() == "png" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !files.isEmpty else { throw NSError(domain: "No PNG frames found", code: 2) }
    let frames = try files.map { try extract($0, includeHands: args.count == 4) }
    try JSONEncoder().encode(frames).write(to: URL(fileURLWithPath: CommandLine.arguments[2]), options: .atomic)
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
