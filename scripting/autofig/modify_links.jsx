app.preferences.setBooleanPreference("ShowExternalJSXWarning", false);
app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

// var oldFolder = "~/<path/to/old/model/directory>";
// var newFolder = "~/<path/to/new/model/directory>";

var oldFolder = "<OLD_FOLDER_PATH>";
var newFolder = "<NEW_FOLDER_PATH>";

var doc = app.activeDocument;
var links = doc.placedItems;

var idxsToRemove = [];  // Track the indexes of the links to remove.

for (var i = 0; i < links.length; i++) {
    var link = links[i];
    if (link.file && link.file.fullName.indexOf(oldFolder) !== -1) {
        var newFilePath = link.file.fullName.replace(oldFolder, newFolder);
        var newFile = new File(newFilePath);
        if (newFile.exists) {
            link.file = newFile;
        } else {
            // Wait to remove the link so as to not mess of the loop
            idxsToRemove.push(i)
        }
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

doc.close();