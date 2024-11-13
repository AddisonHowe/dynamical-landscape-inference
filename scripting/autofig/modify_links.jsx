app.preferences.setBooleanPreference("ShowExternalJSXWarning", false);
app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

// var oldFolder = "~/<path/to/old/model/directory>";
// var newFolder = "~/<path/to/new/model/directory>";

var oldFolder = "<OLD_FOLDER_PATH>";
var newFolder = "<NEW_FOLDER_PATH>";
var logfpath = "<LOGFPATH>";

var doc = app.activeDocument;
var links = doc.placedItems;

var idxsToRemove = [];  // Track the indexes of the links to remove.

var numReplaced = 0;
var numRemoved = 0;
for (var i = 0; i < links.length; i++) {
    var link = links[i];
    try {
        if (link.file && link.file.fullName.indexOf(oldFolder) !== -1) {
            var newFilePath = link.file.fullName.replace(oldFolder, newFolder);
            var newFile = new File(newFilePath);
            if (newFile.exists) {
                link.file = newFile;
                numReplaced++;
            } else {
                // Wait to remove the link so as to not mess up the loop
                idxsToRemove.push(i)
                numRemoved++;
            }
        }
    } catch (e) {
        alert("Caught Error:" + e)
    }
}

// Remove links in reverse order
var nlinks = links.length;
for (var i = 0; i < idxsToRemove.length; i++) {
    link = links[nlinks - i - 1];
    link.remove();
}

// Save ai file and pdf version
doc.save();
var pdfFileName = String(doc.fullName).slice(0, -3) + ".pdf";
var pdfFile = new File(pdfFileName);

var PDFopts = new PDFSaveOptions();
// PDFopts.pDFPreset = "[Smallest File Size]";
PDFopts.pDFPreset = "[Illustrator Default]";
doc.saveAs(pdfFile, PDFopts);

// Log message
var msg = "  Summary:\n\tReplacement folder: " + newFolder + "\n" + 
            "\tTotal number of links: " + nlinks + "\n" +
            "\tReplaced: " + numReplaced + "\n" +
            "\tFailed to replace: " + numRemoved + "\n";
writeToFile(msg, logfpath);

doc.close();


function writeToFile(msg, fpath) {
    try {
        var log = File(fpath);
        log.open('a');
        log.write(msg + '\n');
        log.close()
    } catch (e) {
        $.writeln (e);
    }
};
