// Vision-based OCR helper.
//
// Reads image paths on stdin (one per line), writes one JSON object per line
// to stdout. Bounding boxes are converted from Vision's normalised,
// bottom-left origin into top-left pixel coordinates, so they share a frame
// with the click coordinates recorded by the event stream -- that shared frame
// is what lets the packer answer "what was the label on the thing clicked?".
//
// Language correction is deliberately OFF: this text is file paths, part
// numbers and identifiers, which autocorrect actively damages.
//
// Build: swiftc -O helpers/ocr_mac.swift -o helpers/ocr_mac

import Foundation
import Vision
import CoreGraphics
import ImageIO

func loadImage(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil)
    else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func escape(_ s: String) -> String {
    var out = ""
    for c in s.unicodeScalars {
        switch c {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\t": out += "\\t"
        default:
            if c.value < 0x20 { out += String(format: "\\u%04x", c.value) }
            else { out.unicodeScalars.append(c) }
        }
    }
    return out
}

func ocr(_ path: String) -> String {
    guard let img = loadImage(path) else {
        return "{\"file\":\"\(escape(path))\",\"error\":\"unreadable\"}"
    }
    let w = Double(img.width), h = Double(img.height)
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = false
    req.recognitionLanguages = ["en-US"]

    let handler = VNImageRequestHandler(cgImage: img, options: [:])
    do { try handler.perform([req]) }
    catch {
        return "{\"file\":\"\(escape(path))\",\"error\":\"\(escape(error.localizedDescription))\"}"
    }

    var lines: [String] = []
    for obs in (req.results ?? []) {
        guard let top = obs.topCandidates(1).first else { continue }
        let b = obs.boundingBox                      // normalised, bottom-left
        let x = b.minX * w
        let y = (1.0 - b.maxY) * h                   // flip to top-left origin
        let bw = b.width * w, bh = b.height * h
        lines.append("{\"text\":\"\(escape(top.string))\",\"conf\":\(String(format: "%.3f", top.confidence)),"
            + "\"box\":[\(Int(x.rounded())),\(Int(y.rounded())),\(Int(bw.rounded())),\(Int(bh.rounded()))]}")
    }
    return "{\"file\":\"\(escape(path))\",\"w\":\(img.width),\"h\":\(img.height),"
        + "\"lines\":[\(lines.joined(separator: ","))]}"
}

while let line = readLine(strippingNewline: true) {
    let path = line.trimmingCharacters(in: .whitespaces)
    if path.isEmpty { continue }
    print(ocr(path))
    fflush(stdout)
}
