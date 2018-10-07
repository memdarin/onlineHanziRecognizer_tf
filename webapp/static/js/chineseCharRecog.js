
var strokeFinishedFunction =function(e){

    $('#resultTable').text("");// erases the result table
    charBase64 = $('#signature').jSignature('getData')
    charBase64 = charBase64.replace("data:image/png;base64,", "")
    exportImage(charBase64)
    };

$(document).ready(function() {
	$("#signature").jSignature({width:200,height:200, "background-color":"#FFFFFF", color:"#000000",lineWidth:3});
    $("#signature").bind('change', strokeFinishedFunction)
});

function exportImage(charSignature)
{
	var request = new XMLHttpRequest();
	 request.onreadystatechange = function()
    {
        if (request.readyState == 4 && request.status == 200)
        {
        var json_string = request.responseText
        var result_object = JSON.parse( json_string );
        var chars =result_object.chars
        var probabilities = result_object.probabilities

        var content = '<table border="1"><tr><td>CHARACTERS</td>'
        for (var i = 0; i < chars.length; i++) {
         content += '<td>' + chars[i] + '</td>';
        }
        content += '</tr><tr><td>PROBABILITIES</td>'
         for (var i = 0; i < probabilities.length; i++) {
            content += '<td>' + probabilities[i] + '</td>';
        }
        content += '</tr></table>'
        }
        $('#resultTable').html(content);
    };
    request.open("POST", "http://localhost:5000/addCharImage/", true);
    request.setRequestHeader ('Content-Type', 'application/json');
    request.send(JSON.stringify({
        value: charSignature
    }));
}

function clearChar()
{
    $('#signature').unbind('change')
	$('#signature').jSignature('reset');
	$('#signature').bind('change', strokeFinishedFunction)
	$('#predicted_char').text("");
	$('#resultTable').text("");
}

function importImg(sig)
{
	sig.children("img.imported").remove();
	$("<img class='imported'></img").attr("src",sig.jSignature('getData')).appendTo(sig);
}

function importData(sig)
{
	var dataurl=window.prompt("Paste the exported Image data string here to put it back on this canvas","");
	if ($.trim(dataurl)) {
		sig.jSignature('importData',dataurl);
	}
}
