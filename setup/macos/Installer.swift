// Quake II RTX installer for macOS: a page-for-page counterpart of the
// Windows NSIS installer (setup/setup.nsi, Modern UI 2).
//
//   Welcome -> Install Location -> Choose Components -> Quake II Game Files
//   (full game only) -> Installing -> Completed
//
// build_installer.sh compiles this into "Install Quake II RTX.app" and puts
// the game ("Quake II RTX.app"), the shareware demo files ("shareware/",
// from baseq2/shareware like setup.nsi) and WelcomeImage.bmp in its
// Contents/Resources.
//
// The demo or full game files go into the per-user game directory the engine
// searches (~/.local/share/quake2rtx/baseq2, as on Linux), so the installed
// app bundle stays unmodified and keeps a valid code signature.

import AppKit

let appName = "Quake II RTX"
let setupTitle = "Quake II RTX Setup"

// MARK: - Install state

final class InstallState {
    var installDir = URL(fileURLWithPath: "/Applications")
    var shareware = true                     // Section_Shareware / Section_FullGame radio pair
    var desktopShortcut = true               // Section_DesktopShortcut
    var fullGameDir: URL?                    // $FullGameDir

    let resources = Bundle.main.resourceURL!
    var payloadApp: URL { resources.appendingPathComponent("\(appName).app") }
    var sharewareDir: URL { resources.appendingPathComponent("shareware") }
    var installedApp: URL { installDir.appendingPathComponent("\(appName).app") }

    // The engine's homedir (src/unix/system.c), where game files go.
    static var userGameDir: URL {
        if let xdg = ProcessInfo.processInfo.environment["XDG_DATA_HOME"], !xdg.isEmpty {
            return URL(fileURLWithPath: xdg).appendingPathComponent("quake2rtx")
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".local/share/quake2rtx")
    }
    var dataDir: URL { InstallState.userGameDir.appendingPathComponent("baseq2") }
}

func directorySize(_ url: URL) -> Int64 {
    var total: Int64 = 0
    if let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
        for case let f as URL in e {
            total += Int64((try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
    }
    return total
}

func formatSize(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
}

// FullGamePage_Leave accepts a game folder with baseq2/pak0.pak.
func isGameDir(_ url: URL) -> Bool {
    FileManager.default.fileExists(atPath: url.appendingPathComponent("baseq2/pak0.pak").path)
}

// FullGamePage_Pre: the Steam libraries (macOS Steam keeps them in
// libraryfolders.vdf, like the Windows client).
func detectFullGame() -> URL? {
    let steamApps = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/Steam/steamapps")
    var candidates = [steamApps.appendingPathComponent("common/Quake 2")]
    if let vdf = try? String(contentsOf: steamApps.appendingPathComponent("libraryfolders.vdf"), encoding: .utf8) {
        for line in vdf.split(separator: "\n") {
            let parts = line.split(separator: "\"", omittingEmptySubsequences: true)
                .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            if parts.count == 2, parts[0] == "path" {
                candidates.append(URL(fileURLWithPath: parts[1]).appendingPathComponent("steamapps/common/Quake 2"))
            }
        }
    }
    return candidates.first(where: isGameDir)
}

// MARK: - Small view helpers

func label(_ text: String, bold: Bool = false, size: CGFloat = 12, wraps: Bool = true) -> NSTextField {
    let l = wraps ? NSTextField(wrappingLabelWithString: text) : NSTextField(labelWithString: text)
    l.font = bold ? .boldSystemFont(ofSize: size) : .systemFont(ofSize: size)
    l.translatesAutoresizingMaskIntoConstraints = false
    return l
}

func pin(_ v: NSView, in parent: NSView, _ make: (NSView) -> [NSLayoutConstraint]) {
    v.translatesAutoresizingMaskIntoConstraints = false
    parent.addSubview(v)
    NSLayoutConstraint.activate(make(v))
}

// Checkbox / radio that reports hovering, for the component description box
// (MUI_COMPONENTSPAGE_SMALLDESC).
final class HoverButton: NSButton {
    var onHover: (() -> Void)?
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: bounds, options: [.mouseEnteredAndExited, .activeAlways], owner: self))
    }
    override func mouseEntered(with event: NSEvent) { onHover?() }
}

// MARK: - Pages

protocol WizardPage: AnyObject {
    var view: NSView { get }
    var headerTitle: String { get }       // MUI_PAGE_HEADER_TEXT
    var headerSubtitle: String { get }
    var isWelcome: Bool { get }
    func willShow()
    func canLeave() -> Bool               // the page's _Leave function
}

extension WizardPage {
    var isWelcome: Bool { false }
    func willShow() {}
    func canLeave() -> Bool { true }
}

// MUI_PAGE_WELCOME: the banner on the left, the text on white.
final class WelcomePage: WizardPage {
    let view = NSView()
    let headerTitle = ""
    let headerSubtitle = ""
    var isWelcome: Bool { true }
    var onUninstall: (() -> Void)?

    init(state: InstallState) {
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.white.cgColor

        let banner = NSImageView(image: NSImage(contentsOf: state.resources.appendingPathComponent("WelcomeImage.bmp")) ?? NSImage())
        banner.imageScaling = .scaleAxesIndependently
        pin(banner, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor),
                                 $0.topAnchor.constraint(equalTo: self.view.topAnchor),
                                 $0.bottomAnchor.constraint(equalTo: self.view.bottomAnchor),
                                 $0.widthAnchor.constraint(equalToConstant: 164)] }

        let title = label("Welcome to \(appName) Setup", bold: true, size: 16)
        title.textColor = .black
        let body = label("Setup will guide you through the installation of \(appName).\n\n" +
                         "It is recommended that you close all other applications before starting Setup. " +
                         "This will make it possible to update relevant system files without having to reboot your computer.\n\n" +
                         "Click Next to continue.")
        body.textColor = .black
        let uninstall = NSButton(title: "Uninstall \(appName)…", target: nil, action: nil)
        uninstall.bezelStyle = .inline
        uninstall.target = self
        uninstall.action = #selector(uninstallClicked)

        for (v, top) in [(title as NSView, 20.0), (body, 70.0)] {
            pin(v, in: view) { [$0.leadingAnchor.constraint(equalTo: banner.trailingAnchor, constant: 18),
                                $0.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -18),
                                $0.topAnchor.constraint(equalTo: self.view.topAnchor, constant: top)] }
        }
        pin(uninstall, in: view) { [$0.leadingAnchor.constraint(equalTo: banner.trailingAnchor, constant: 18),
                                    $0.bottomAnchor.constraint(equalTo: self.view.bottomAnchor, constant: -14)] }
    }

    @objc func uninstallClicked() { onUninstall?() }
}

// MUI_PAGE_DIRECTORY
final class DirectoryPage: WizardPage {
    let view = NSView()
    let headerTitle = "Choose Install Location"
    let headerSubtitle = "Choose the folder in which to install \(appName)."
    let state: InstallState
    let path = NSTextField()
    let required: NSTextField
    let available = label("", wraps: false)

    init(state: InstallState) {
        self.state = state
        required = label("Space required: \(formatSize(directorySize(state.payloadApp) + directorySize(state.sharewareDir)))", wraps: false)
        let top = label("Setup will install \(appName) in the following folder. To install in a different folder, click Browse and select another folder. Click Next to continue.\n\n" +
                        "Installing into a folder with an existing installation of Quake II is NOT recommended.")
        let box = NSBox()
        box.title = "Destination Folder"
        path.stringValue = state.installDir.path
        path.isEditable = false
        let browse = NSButton(title: "Browse…", target: nil, action: nil)
        browse.target = self
        browse.action = #selector(browseClicked)

        pin(top, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                              $0.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -20),
                              $0.topAnchor.constraint(equalTo: self.view.topAnchor, constant: 14)] }
        pin(box, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                              $0.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -20),
                              $0.topAnchor.constraint(equalTo: top.bottomAnchor, constant: 16),
                              $0.heightAnchor.constraint(equalToConstant: 64)] }
        let content = box.contentView!
        pin(path, in: content) { [$0.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 10),
                                  $0.centerYAnchor.constraint(equalTo: content.centerYAnchor)] }
        pin(browse, in: content) { [$0.leadingAnchor.constraint(equalTo: self.path.trailingAnchor, constant: 8),
                                    $0.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -10),
                                    $0.centerYAnchor.constraint(equalTo: content.centerYAnchor)] }
        pin(required, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                                   $0.topAnchor.constraint(equalTo: box.bottomAnchor, constant: 16)] }
        pin(available, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                                    $0.topAnchor.constraint(equalTo: self.required.bottomAnchor, constant: 4)] }
        updateAvailable()
    }

    func updateAvailable() {
        let free = (try? state.installDir.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage ?? 0
        available.stringValue = "Space available: \(formatSize(free))"
    }

    @objc func browseClicked() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.directoryURL = state.installDir
        panel.message = "Choose the folder in which to install \(appName)."
        if panel.runModal() == .OK, let url = panel.url {
            state.installDir = url
            path.stringValue = url.path
            updateAvailable()
        }
    }
}

// MUI_PAGE_COMPONENTS
final class ComponentsPage: WizardPage {
    let view = NSView()
    let headerTitle = "Choose Components"
    let headerSubtitle = "Choose which features of \(appName) you want to install."
    let state: InstallState
    let engine = HoverButton(checkboxWithTitle: "Engine Files (Required)", target: nil, action: nil)
    let shareware = HoverButton(radioButtonWithTitle: "Quake II Shareware Demo", target: nil, action: nil)
    let fullGame = HoverButton(radioButtonWithTitle: "Quake II Full Game", target: nil, action: nil)
    let shortcut = HoverButton(checkboxWithTitle: "Desktop Shortcut", target: nil, action: nil)
    let desc = label("Position your mouse over a component to see its description.", size: 11)

    init(state: InstallState) {
        self.state = state
        let top = label("Check the components you want to install and uncheck the components you don't want to install. Click Next to continue.")
        engine.state = .on
        engine.isEnabled = false
        shareware.state = state.shareware ? .on : .off
        fullGame.state = state.shareware ? .off : .on
        shortcut.state = state.desktopShortcut ? .on : .off
        for b in [shareware, fullGame, shortcut] {
            b.target = self
            b.action = #selector(changed)
        }
        // MUI_DESCRIPTION_TEXT
        let descriptions: [(HoverButton, String)] = [
            (engine, "Executable and media files for \(appName)"),
            (shareware, "Install a copy of the Quake II Shareware Demo"),
            (fullGame, "Locate and copy the media files for the full game"),
            (shortcut, "Place a shortcut for the game onto the Desktop"),
        ]
        for (b, text) in descriptions {
            b.onHover = { [weak self] in self?.desc.stringValue = text }
        }

        let list = NSStackView(views: [engine, shareware, fullGame, shortcut])
        list.orientation = .vertical
        list.alignment = .leading
        list.spacing = 8
        let descBox = NSBox()
        descBox.title = "Description"

        pin(top, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                              $0.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -20),
                              $0.topAnchor.constraint(equalTo: self.view.topAnchor, constant: 14)] }
        let select = label("Select components to install:", wraps: false)
        pin(select, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                                 $0.topAnchor.constraint(equalTo: top.bottomAnchor, constant: 18)] }
        pin(list, in: view) { [$0.leadingAnchor.constraint(equalTo: select.trailingAnchor, constant: 16),
                               $0.topAnchor.constraint(equalTo: select.topAnchor)] }
        pin(descBox, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                                  $0.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -20),
                                  $0.topAnchor.constraint(equalTo: list.bottomAnchor, constant: 18),
                                  $0.heightAnchor.constraint(equalToConstant: 60)] }
        let content = descBox.contentView!
        pin(desc, in: content) { [$0.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 8),
                                  $0.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -8),
                                  $0.topAnchor.constraint(equalTo: content.topAnchor, constant: 4)] }
    }

    // .onSelChange: shareware and full game are a radio pair.
    @objc func changed(_ sender: NSButton) {
        if sender === shareware { fullGame.state = .off }
        if sender === fullGame { shareware.state = .off }
        state.shareware = shareware.state == .on
        state.desktopShortcut = shortcut.state == .on
    }
}

// The custom "Quake II Game Files" directory page (FullGamePage_Pre/_Leave).
final class FullGamePage: WizardPage {
    let view = NSView()
    let headerTitle = "Quake II Game Files"
    let headerSubtitle = ""
    let state: InstallState
    let path = NSTextField()

    init(state: InstallState) {
        self.state = state
        let top = label("Choose the folder where the Quake II game files are located. The installer will copy the necessary files to the \(appName) install location.")
        let box = NSBox()
        box.title = "Folder with the Quake II game files (the folder that contains baseq2)"
        path.isEditable = false
        let browse = NSButton(title: "Browse…", target: nil, action: nil)
        browse.target = self
        browse.action = #selector(browseClicked)

        pin(top, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                              $0.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -20),
                              $0.topAnchor.constraint(equalTo: self.view.topAnchor, constant: 14)] }
        pin(box, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                              $0.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -20),
                              $0.topAnchor.constraint(equalTo: top.bottomAnchor, constant: 16),
                              $0.heightAnchor.constraint(equalToConstant: 64)] }
        let content = box.contentView!
        pin(path, in: content) { [$0.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 10),
                                  $0.centerYAnchor.constraint(equalTo: content.centerYAnchor)] }
        pin(browse, in: content) { [$0.leadingAnchor.constraint(equalTo: self.path.trailingAnchor, constant: 8),
                                    $0.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -10),
                                    $0.centerYAnchor.constraint(equalTo: content.centerYAnchor)] }
    }

    func willShow() {
        if state.fullGameDir == nil { state.fullGameDir = detectFullGame() }
        path.stringValue = state.fullGameDir?.path ?? ""
    }

    func canLeave() -> Bool {
        if let dir = state.fullGameDir, isGameDir(dir) { return true }
        let alert = NSAlert()
        alert.messageText = "Game files (baseq2/pak*.pak) are not found in the specified location. Please specify the correct location."
        alert.runModal()
        return false
    }

    @objc func browseClicked() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.directoryURL = state.fullGameDir
        if panel.runModal() == .OK, var url = panel.url {
            // Choosing the baseq2 folder itself is fine too.
            if !isGameDir(url), isGameDir(url.deletingLastPathComponent()) { url = url.deletingLastPathComponent() }
            state.fullGameDir = url
            path.stringValue = url.path
        }
    }
}

// MUI_PAGE_INSTFILES: progress bar and the details list.
final class InstallPage: WizardPage {
    let view = NSView()
    var headerTitle = "Installing"
    var headerSubtitle = "Please wait while \(appName) is being installed."
    let status = label("", wraps: false)
    let progress = NSProgressIndicator()
    let details = NSTextView()

    init() {
        progress.isIndeterminate = false
        progress.minValue = 0
        progress.maxValue = 1
        status.lineBreakMode = .byTruncatingMiddle
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        details.isEditable = false
        details.font = .monospacedSystemFont(ofSize: 10, weight: .regular)
        details.autoresizingMask = [.width]
        scroll.documentView = details

        pin(status, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                                 $0.trailingAnchor.constraint(lessThanOrEqualTo: self.view.trailingAnchor, constant: -20),
                                 $0.topAnchor.constraint(equalTo: self.view.topAnchor, constant: 14)] }
        pin(progress, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                                   $0.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -20),
                                   $0.topAnchor.constraint(equalTo: self.status.bottomAnchor, constant: 8)] }
        pin(scroll, in: view) { [$0.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                                 $0.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -20),
                                 $0.topAnchor.constraint(equalTo: self.progress.bottomAnchor, constant: 10),
                                 $0.bottomAnchor.constraint(equalTo: self.view.bottomAnchor, constant: -10)] }
    }

    func log(_ line: String) {
        status.stringValue = line
        details.textStorage?.append(NSAttributedString(string: line + "\n", attributes: [.font: details.font!, .foregroundColor: NSColor.textColor]))
        details.scrollToEndOfDocument(nil)
    }
}

// MARK: - Installation

enum InstallError: LocalizedError {
    case failed(String)
    var errorDescription: String? { if case .failed(let s) = self { return s }; return nil }
}

@discardableResult
func run(_ tool: String, _ args: [String]) throws -> Int32 {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: tool)
    p.arguments = args
    try p.run()
    p.waitUntilExit()
    return p.terminationStatus
}

// Runs a shell command as an administrator (Authorization prompt).
func runAsAdmin(_ shell: String) throws {
    let escaped = shell.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
    if try run("/usr/bin/osascript", ["-e", "do shell script \"\(escaped)\" with administrator privileges"]) != 0 {
        throw InstallError.failed("The installation was cancelled or the administrator password was not accepted.")
    }
}

func q(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }

final class Installation {
    let state: InstallState
    let page: InstallPage
    private var done: Int64 = 0
    private var total: Int64 = 1

    init(state: InstallState, page: InstallPage) {
        self.state = state
        self.page = page
    }

    private func ui(_ f: @escaping () -> Void) { DispatchQueue.main.async(execute: f) }
    private func log(_ s: String) { ui { self.page.log(s) } }
    private func advance(_ bytes: Int64) {
        done += bytes
        let v = Double(done) / Double(max(total, 1))
        ui { self.page.progress.doubleValue = v }
    }

    // Copies a tree file by file for the progress bar and the details list.
    private func copyTree(_ src: URL, to dst: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: dst, withIntermediateDirectories: true)
        let keys: [URLResourceKey] = [.isDirectoryKey, .fileSizeKey, .isSymbolicLinkKey]
        guard let e = fm.enumerator(at: src, includingPropertiesForKeys: keys) else { return }
        let base = src.standardizedFileURL.path
        for case let f as URL in e {
            let rel = String(f.standardizedFileURL.path.dropFirst(base.count + 1))
            let out = dst.appendingPathComponent(rel)
            let v = try f.resourceValues(forKeys: Set(keys))
            if v.isDirectory == true && v.isSymbolicLink != true {
                try fm.createDirectory(at: out, withIntermediateDirectories: true)
                continue
            }
            log("Extract: \(rel)")
            if fm.fileExists(atPath: out.path) { try fm.removeItem(at: out) }
            try fm.copyItem(at: f, to: out)
            advance(Int64(v.fileSize ?? 0))
        }
    }

    func run() throws {
        let fm = FileManager.default
        total = directorySize(state.payloadApp)
        if state.shareware { total += directorySize(state.sharewareDir) }
        else if let g = state.fullGameDir { total += directorySize(g.appendingPathComponent("baseq2")) }

        // Section_Game
        log("Output folder: \(state.installDir.path)")
        let app = state.installedApp
        do {
            if fm.fileExists(atPath: app.path) { try fm.removeItem(at: app) }
            try copyTree(state.payloadApp, to: app)
        } catch {
            // e.g. a folder only an administrator may write to
            log("Copying \(appName).app as administrator…")
            try runAsAdmin("rm -rf \(q(app.path)) && ditto \(q(state.payloadApp.path)) \(q(app.path))")
            advance(directorySize(state.payloadApp))
        }
        _ = try? Installation.runQuiet("/usr/bin/xattr", ["-dr", "com.apple.quarantine", app.path])

        // Section_Shareware / Section_FullGame
        let data = state.dataDir
        try fm.createDirectory(at: data, withIntermediateDirectories: true)
        log("Output folder: \(data.path)")
        if state.shareware {
            let target = data.appendingPathComponent("pak0.pak")
            if fm.fileExists(atPath: data.appendingPathComponent("pak1.pak").path) {
                log("Game files of the full game are already installed; skipping the shareware demo.")
            } else {
                log("Extract: pak0.pak")
                if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
                try fm.copyItem(at: state.sharewareDir.appendingPathComponent("pak0.pak"), to: target)
                advance(Int64((try? target.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0))
                try copyTree(state.sharewareDir.appendingPathComponent("players"), to: data.appendingPathComponent("players"))
            }
        } else if let game = state.fullGameDir {
            let base = game.appendingPathComponent("baseq2")
            let paks = (try fm.contentsOfDirectory(atPath: base.path)).filter { $0.lowercased().hasPrefix("pak") && $0.lowercased().hasSuffix(".pak") }
            for pak in paks.sorted() {
                log("Copy: \(pak)")
                let out = data.appendingPathComponent(pak)
                if fm.fileExists(atPath: out.path) { try fm.removeItem(at: out) }
                try fm.copyItem(at: base.appendingPathComponent(pak), to: out)
                advance(Int64((try? out.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0))
            }
            for dir in ["players", "video", "music"] where fm.fileExists(atPath: base.appendingPathComponent(dir).path) {
                try copyTree(base.appendingPathComponent(dir), to: data.appendingPathComponent(dir))
            }
            // The GOG release keeps the music next to baseq2.
            if fm.fileExists(atPath: game.appendingPathComponent("music").path) {
                try copyTree(game.appendingPathComponent("music"), to: data.appendingPathComponent("music"))
            }
        }

        // Section_DesktopShortcut: a Finder alias.
        if state.desktopShortcut {
            let alias = fm.homeDirectoryForCurrentUser.appendingPathComponent("Desktop/\(appName)")
            log("Create shortcut: \(alias.path)")
            try? fm.removeItem(at: alias)
            if let bookmark = try? app.bookmarkData(options: .suitableForBookmarkFile, includingResourceValuesForKeys: nil, relativeTo: nil) {
                try? URL.writeBookmarkData(bookmark, to: alias)
            }
        }
        log("Completed")
        ui { self.page.progress.doubleValue = 1 }
    }

    @discardableResult
    static func runQuiet(_ tool: String, _ args: [String]) throws -> Int32 {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: tool)
        p.arguments = args
        p.standardError = FileHandle.nullDevice
        try p.run()
        p.waitUntilExit()
        return p.terminationStatus
    }
}

// The "Uninstall" section: the app, the desktop shortcut and, if asked, the
// installed game files, saves and settings.
func uninstall(window: NSWindow) {
    let fm = FileManager.default
    let apps = ["/Applications", fm.homeDirectoryForCurrentUser.appendingPathComponent("Applications").path]
        .map { URL(fileURLWithPath: $0).appendingPathComponent("\(appName).app") }
        .filter { fm.fileExists(atPath: $0.path) }

    let alert = NSAlert()
    if apps.isEmpty {
        alert.messageText = "\(appName) was not found in Applications."
        alert.informativeText = "To remove it from another folder, drag it to the Trash."
        alert.beginSheetModal(for: window)
        return
    }
    alert.messageText = "Remove \(appName)?"
    alert.informativeText = apps.map(\.path).joined(separator: "\n") +
        "\n\nThe installed game files, saved games and settings in \(InstallState.userGameDir.path) can be kept or deleted."
    alert.addButton(withTitle: "Remove, Keep Game Files")
    alert.addButton(withTitle: "Remove Everything")
    alert.addButton(withTitle: "Cancel")
    alert.alertStyle = .warning
    alert.beginSheetModal(for: window) { response in
        guard response != .alertThirdButtonReturn else { return }
        for app in apps {
            if (try? fm.removeItem(at: app)) == nil {
                try? runAsAdmin("rm -rf \(q(app.path))")
            }
        }
        try? fm.removeItem(at: fm.homeDirectoryForCurrentUser.appendingPathComponent("Desktop/\(appName)"))
        if response == .alertSecondButtonReturn {
            try? fm.removeItem(at: InstallState.userGameDir)
        }
        let done = NSAlert()
        done.messageText = "\(appName) was successfully removed from your computer."
        done.beginSheetModal(for: window) { _ in NSApp.terminate(nil) }
    }
}

// MARK: - Wizard window

final class Wizard: NSObject, NSWindowDelegate {
    let state = InstallState()
    let window: NSWindow
    let header = NSView()
    let headerTitle = label("", bold: true, size: 13, wraps: false)
    let headerSubtitle = label("", wraps: false)
    let pageArea = NSView()
    let headerLine = NSBox()
    let back = NSButton(title: "< Back", target: nil, action: nil)
    let next = NSButton(title: "Next >", target: nil, action: nil)
    let cancel = NSButton(title: "Cancel", target: nil, action: nil)

    var pages: [WizardPage] = []
    var index = 0
    let installPage = InstallPage()
    var installing = false
    var finished = false

    override init() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 500, height: 390),
                          styleMask: [.titled, .closable], backing: .buffered, defer: false)
        super.init()
        window.title = setupTitle
        window.delegate = self

        let welcome = WelcomePage(state: state)
        welcome.onUninstall = { [unowned self] in uninstall(window: self.window) }
        pages = [welcome, DirectoryPage(state: state), ComponentsPage(state: state), FullGamePage(state: state), installPage]

        let root = window.contentView!
        // White header band with the title, subtitle and icon (MUI header).
        header.wantsLayer = true
        header.layer?.backgroundColor = NSColor.white.cgColor
        headerTitle.textColor = .black
        headerSubtitle.textColor = .black
        let icon = NSImageView(image: NSImage(contentsOf: state.resources.appendingPathComponent("q2rtx.icns")) ?? NSApp.applicationIconImage)
        pin(header, in: root) { [$0.leadingAnchor.constraint(equalTo: root.leadingAnchor),
                                 $0.trailingAnchor.constraint(equalTo: root.trailingAnchor),
                                 $0.topAnchor.constraint(equalTo: root.topAnchor),
                                 $0.heightAnchor.constraint(equalToConstant: 58)] }
        pin(headerTitle, in: header) { [$0.leadingAnchor.constraint(equalTo: self.header.leadingAnchor, constant: 16),
                                        $0.topAnchor.constraint(equalTo: self.header.topAnchor, constant: 11)] }
        pin(headerSubtitle, in: header) { [$0.leadingAnchor.constraint(equalTo: self.header.leadingAnchor, constant: 26),
                                           $0.topAnchor.constraint(equalTo: self.headerTitle.bottomAnchor, constant: 4)] }
        pin(icon, in: header) { [$0.trailingAnchor.constraint(equalTo: self.header.trailingAnchor, constant: -10),
                                 $0.centerYAnchor.constraint(equalTo: self.header.centerYAnchor),
                                 $0.widthAnchor.constraint(equalToConstant: 44), $0.heightAnchor.constraint(equalToConstant: 44)] }
        headerLine.boxType = .separator
        pin(headerLine, in: root) { [$0.leadingAnchor.constraint(equalTo: root.leadingAnchor),
                                     $0.trailingAnchor.constraint(equalTo: root.trailingAnchor),
                                     $0.topAnchor.constraint(equalTo: self.header.bottomAnchor)] }

        // Footer: branding text over a separator, then the buttons.
        let footerLine = NSBox()
        footerLine.boxType = .separator
        let brand = label(setupTitle, size: 10, wraps: false)
        brand.textColor = .tertiaryLabelColor
        for b in [back, next, cancel] { b.bezelStyle = .rounded; b.target = self }
        back.action = #selector(goBack)
        next.action = #selector(goNext)
        cancel.action = #selector(cancelClicked)
        next.keyEquivalent = "\r"
        pin(cancel, in: root) { [$0.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -12),
                                 $0.bottomAnchor.constraint(equalTo: root.bottomAnchor, constant: -10),
                                 $0.widthAnchor.constraint(greaterThanOrEqualToConstant: 80)] }
        pin(next, in: root) { [$0.trailingAnchor.constraint(equalTo: self.cancel.leadingAnchor, constant: -12),
                               $0.centerYAnchor.constraint(equalTo: self.cancel.centerYAnchor),
                               $0.widthAnchor.constraint(greaterThanOrEqualToConstant: 80)] }
        pin(back, in: root) { [$0.trailingAnchor.constraint(equalTo: self.next.leadingAnchor),
                               $0.centerYAnchor.constraint(equalTo: self.cancel.centerYAnchor),
                               $0.widthAnchor.constraint(greaterThanOrEqualToConstant: 80)] }
        pin(brand, in: root) { [$0.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: 8),
                                $0.bottomAnchor.constraint(equalTo: self.cancel.topAnchor, constant: -4)] }
        pin(footerLine, in: root) { [$0.leadingAnchor.constraint(equalTo: brand.trailingAnchor, constant: 6),
                                     $0.trailingAnchor.constraint(equalTo: root.trailingAnchor),
                                     $0.centerYAnchor.constraint(equalTo: brand.centerYAnchor)] }

        pin(pageArea, in: root) { [$0.leadingAnchor.constraint(equalTo: root.leadingAnchor),
                                   $0.trailingAnchor.constraint(equalTo: root.trailingAnchor),
                                   $0.topAnchor.constraint(equalTo: root.topAnchor),
                                   $0.bottomAnchor.constraint(equalTo: footerLine.topAnchor, constant: -1)] }
        showPage()
        window.center()
    }

    // Pages that apply to the current choices (FullGamePage_Pre aborts
    // when the full game is not selected).
    func applies(_ p: WizardPage) -> Bool { !(p is FullGamePage) || !state.shareware }

    func showPage() {
        let page = pages[index]
        pageArea.subviews.forEach { $0.removeFromSuperview() }
        let welcome = page.isWelcome
        header.isHidden = welcome
        headerLine.isHidden = welcome
        headerTitle.stringValue = page.headerTitle
        headerSubtitle.stringValue = page.headerSubtitle
        page.willShow()
        pin(page.view, in: pageArea) { [$0.leadingAnchor.constraint(equalTo: self.pageArea.leadingAnchor),
                                        $0.trailingAnchor.constraint(equalTo: self.pageArea.trailingAnchor),
                                        $0.topAnchor.constraint(equalTo: self.pageArea.topAnchor, constant: welcome ? 0 : 59),
                                        $0.bottomAnchor.constraint(equalTo: self.pageArea.bottomAnchor)] }
        back.isHidden = index == 0 || page === installPage
        let nextPage = pages[(index + 1)...].first(where: applies)
        next.title = nextPage === installPage ? "Install" : "Next >"
        next.isEnabled = true
        if page === installPage { next.isEnabled = false; cancel.isEnabled = false }
    }

    @objc func goBack() {
        guard let i = pages[..<index].lastIndex(where: applies) else { return }
        index = i
        showPage()
    }

    @objc func goNext() {
        if finished { window.close(); return }
        guard pages[index].canLeave(), let i = pages[(index + 1)...].firstIndex(where: applies) else { return }
        index = i
        showPage()
        if pages[index] === installPage { startInstall() }
    }

    func startInstall() {
        installing = true
        let job = Installation(state: state, page: installPage)
        DispatchQueue.global(qos: .userInitiated).async {
            var failure: Error?
            do { try job.run() } catch { failure = error }
            DispatchQueue.main.async { self.installFinished(failure) }
        }
    }

    func installFinished(_ error: Error?) {
        installing = false
        finished = true
        if let error = error {
            installPage.log("Error: \(error.localizedDescription)")
            installPage.headerTitle = "Installation Aborted"
            installPage.headerSubtitle = "Setup was not completed successfully."
        } else {
            installPage.headerTitle = "Installation Complete"
            installPage.headerSubtitle = "Setup was completed successfully."
        }
        headerTitle.stringValue = installPage.headerTitle
        headerSubtitle.stringValue = installPage.headerSubtitle
        next.title = "Close"
        next.isEnabled = true
        cancel.isEnabled = false
        if error == nil {
            // Optional extra over the NSIS installer: start the game now.
            let play = NSButton(checkboxWithTitle: "Run \(appName)", target: nil, action: nil)
            play.state = .on
            pin(play, in: window.contentView!) { [$0.leadingAnchor.constraint(equalTo: self.window.contentView!.leadingAnchor, constant: 20),
                                                  $0.centerYAnchor.constraint(equalTo: self.cancel.centerYAnchor)] }
            playCheckbox = play
        }
    }
    var playCheckbox: NSButton?

    @objc func cancelClicked() { window.performClose(nil) }

    // MUI_ABORTWARNING
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if installing { return false }
        if finished { return true }
        let alert = NSAlert()
        alert.messageText = "Are you sure you want to quit \(appName) Setup?"
        alert.addButton(withTitle: "Yes")
        alert.addButton(withTitle: "No")
        return alert.runModal() == .alertFirstButtonReturn
    }

    func windowWillClose(_ notification: Notification) {
        if finished, playCheckbox?.state == .on {
            NSWorkspace.shared.openApplication(at: state.installedApp, configuration: NSWorkspace.OpenConfiguration())
        }
        NSApp.terminate(nil)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var wizard: Wizard?
    func applicationDidFinishLaunching(_ notification: Notification) {
        wizard = Wizard()
        wizard?.window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
