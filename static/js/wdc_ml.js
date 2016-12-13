//Declare Global variables
var submit = 0; 
var trainingFileTxt = "" ;
var trainingFileSize = "";
var scoringFileTxt = "";
var scoringFileSize = "";
var algorithmChosen;
var L1L2;
var GridSearch;
var numbTrees;
var splitCriteria;
var fieldNamesArray = [];
var fieldTypesArray = [];

//Onload call init
window.onload = init;


function wait(ms){
    var start = new Date().getTime();
    var end = start;
    while(end < start + ms) {
        end = new Date().getTime();
    }
}

function validateUploadForm(trainingFile, scoringFile, algorithm) {
    var errormsg = "";
    
    var isValid = true;
    algorithm = algorithm.trim(); 
    if (trainingFile != null) {trainingFileTxt = trainingFile.name; 
                               trainingFileSize = trainingFile.size;
                               var trainFileExtension = trainingFileTxt.substr(trainingFileTxt.lastIndexOf('.')+1);
        }
    if (scoringFile != null)  {scoringFileTxt = scoringFile.name; 
                               scoringFileSize = scoringFile.size;
                               var scoreFileExtension = scoringFileTxt.substr(scoringFileTxt.lastIndexOf('.')+1);
        }
    
    var allowedFileTypes = ['csv', 'txt', 'xlsx'];
    
    

    //"Algorithms" is the default value. Indicates that the user has not selected an value 
    if (algorithm == "Algorithms" ) {   
        errormsg += "Please choose an algorithm";
        isValid = false;
    }
    
    else if (trainingFileTxt == "" ) {
        errormsg += "Please choose a training file to upload";
        isValid = false;
    }
     
    else if (scoringFileTxt == "" && algorithm != 'Clustering(K-means)') {
        errormsg += "Please choose a file to score";
        isValid = false;
    }
    
     else if (algorithm == 'Clustering(K-means)' && allowedFileTypes.indexOf(trainFileExtension) == -1 ) {      // > -1 if in the array
        errormsg += "Only .csv, .txt or .xlsx files are currently supported";
        isValid = false;
    }

     else if (algorithm != 'Clustering(K-means)' && (allowedFileTypes.indexOf(trainFileExtension) == -1 || allowedFileTypes.indexOf(scoreFileExtension) == -1)) {   
        errormsg += "Only .csv, .txt or .xlsx files are currently supported";
        isValid = false;
     }
     
     // if trainfile or scorefile is larger than 10Mb then do not upload (you may change this to better suit your environment) 
    else if (trainingFileSize > 10000000 || scoringFileSize > 10000000 ) {
        errormsg += "maximum file size to upload is currently 10 MB";
        isValid = false;
    }
    
    if (isValid == false) {
            return errormsg;
    }
    
    else  { // if isValid then return True 
            return true;  
    } 
}

function validateProcessForm(trees) {
    var errormsg = "";
    var isValid = true;
    // if user has selected more than 25 trees
    if (trees > 25 ) {
        errormsg += "maximum of 25 trees in the Random Forest";
        isValid = false;
    }
     
    if (isValid == false) {
        return errormsg;
    }
    
    else  { // if isValid then return True 
        return true;  
    } 
}


function getTableauConnData(algorithmChosen){
    if (algorithmChosen == 'Clustering(K-means)') {
          return JSON.stringify({'TrainfilNamn': trainingFileTxt, 'ScorefilNamn': scoringFileTxt, 
         'fnArray': fieldNamesArray, 'ftArray': fieldTypesArray, 'algoChosen':algorithmChosen});
    } else if (algorithmChosen == 'Logistic Regression') {
         return JSON.stringify({'TrainfilNamn': trainingFileTxt, 'ScorefilNamn': scoringFileTxt, 
         'fnArray': fieldNamesArray, 'ftArray': fieldTypesArray, 'algoChosen':algorithmChosen,
         'regularisationParam1': L1L2, 'regularisationParam2':GridSearch});
    } else if (algorithmChosen == 'Random Forest') {
          return JSON.stringify({'TrainfilNamn': trainingFileTxt, 'ScorefilNamn': scoringFileTxt, 
         'fnArray': fieldNamesArray, 'ftArray': fieldTypesArray, 'algoChosen':algorithmChosen, 'numbTrees':numbTrees, 'splitCriteria':splitCriteria});
    } else if (algorithmChosen == 'OLS Linear Regression') {
          return JSON.stringify({'TrainfilNamn': trainingFileTxt, 'ScorefilNamn': scoringFileTxt, 
         'fnArray': fieldNamesArray, 'ftArray': fieldTypesArray, 'algoChosen':algorithmChosen});
    } else {
          return false; 
    }
}

function spinner() {
    $("#loadspin").show();  
}

function fileUpload() {
    
    var  uploadFunc = "";
    var trainingFile = $('input[type=file]')[0].files[0];
    var scoringFile = $('input[type=file]')[1].files[0];

    algorithmChosen = $('.algoStatus').text();
    numbTrees = $("#numbtrees2").val();
    
    //validate the form by calling the validateUploadForm function
    var validForm = validateUploadForm (trainingFile, scoringFile, algorithmChosen);
    if (validForm == true) { 

        algorithmChosen = algorithmChosen.trim();
     
        if (algorithmChosen == 'Clustering(K-means)') {
            uploadFunc = "Kmeans";
        } else if (algorithmChosen == 'Logistic Regression') {
             uploadFunc = "Classification";
        } else if (algorithmChosen == 'Random Forest') {
            uploadFunc = "Classification";
        } else if (algorithmChosen == 'OLS Linear Regression') {
            uploadFunc = "OLS";
        } else {
            uploadFunc ="Error";
        }
        
        //get the selected files from the form 
        var form_data = new FormData();
        form_data.append('trainfile', $('input[type=file]')[0].files[0]);
        form_data.append('scorefile', $('input[type=file]')[1].files[0]);

        //ajax request for file upload
        $.ajax({
            type: 'POST',
            url: '/uploadFile' + uploadFunc,
            data: form_data,
            dataType: "json",       //Converts the json String into a true json object
            contentType: false,
            cache: false,
            processData: false,
            async: true,  //false
            beforeSend: function() { 
                        spinner();
                        },
            success: function(data) {                
                     fieldNamesArray = [];
                     fieldTypesArray = [];
                     if (data[0] == 'error') {
                         $("#loadspin").hide();  
                         $("#errorInfo").fadeIn(500);
                         $("#errorInfo").html("<i class='fa fa-exclamation-triangle fa-2x'" + 
                         'aria-hidden="true"></i> <strong>&nbsp; Error:</strong>' + data[1]);
                     } else {           
                         for (var i = 0; i < data[0].length; i++) {
                         fieldNamesArray.push(data[0][i]);      //Push into the global array
                         }
                         for (var i = 0; i < data[1].length; i++) {
                            fieldTypesArray.push(data[1][i]);     //Push into the global array
                         }
                                        
                         console.log('Success!');
                         $(".uploadparameters").hide(); 
                         $("#errorInfo").hide();
                         $("#successInfo").fadeIn(800);
                         $("#processFilebtn").fadeIn(800);
                         $("#cancelbtn").fadeIn(800);
                        if (algorithmChosen == 'Logistic Regression') {
                             $(".LogisticParameter").fadeIn(800); 
                             $("#parameterInfo").fadeIn(800);
                             }
                        else if (algorithmChosen == 'Random Forest') {
                            $(".RandomForestParameter").fadeIn(800); 
                            $("#parameterInfo").fadeIn(800);
                             }
                         
                         $("#loadspin").hide();                 
                      }         
                   },  //end of success function
         // complete: function() {
                                 // },
            error: function(data) { 
                       console.log("Error!");
                       $("#errorInfo").fadeIn(500);
                       $("#loadspin").hide(); 
    
                    }   
        }); //end of ajax   

    } 
    //if form is not valid enter errormsg text in the html 
    else {
        $("#errorInfo").fadeIn(500);
        $("#errorInfo").html("<i class='fa fa-exclamation-triangle fa-2x'" + 
        'aria-hidden="true"></i> <strong>&nbsp; Error:</strong> ' + validForm);
    }
} //end of fileUpload function

function bindingButtons() {
    //algorithm button change text click event
    $('.algoclass li > a').click(function(e){
        $('.algoStatus').html($(this).text() + ' <span class="caret"></span>');
    });
    
    //If dropdown value is 'Clustering(K-means)' then we hide the trainFile button and label
    $('.algoclass li > a').click(function(e){
        dropdownVal = $('.algoStatus').text();
        if (dropdownVal.trim() == 'Clustering(K-means)') {
            $("#scoreFilefg").hide();      
        }
        else {
            $("#scoreFilefg").fadeIn(250); 
        }
        
    });
    
    //bind cancelProcessData() to cancelBtn click event
    $('#cancelBtn').click(function(){
        cancelProcessData();
    });
    //bind processData() to processBtn click event
    $('#processBtn').click(function(){
        processData();
    });
    //bind fileUpload() to fileUploadBtnclick event
    $('#fileUploadBtn').click(function(){
        fileUpload();
    });
    
} //end bindingButtons functions

function init() {

    // Call function for binding the buttons
    bindingButtons();

    //Hide the following divs
    $("#successInfo").hide();
    $("#parameterInfo").hide(); 
    $("#errorInfo").hide();
    $("#processFilebtn").hide();
    $("#cancelbtn").hide(); 
    $("#loadspin").hide(); 
    $(".LogisticParameter").hide(); 
    $(".RandomForestParameter").hide(); 


    var myConnector = tableau.makeConnector();
    myConnector.getColumnHeaders = function() {
                        if (submit === 0) {

                            var tableauConnData2 = tableau.connectionData;
                            var tableauConnDataObj2 = JSON.parse(tableauConnData2);
                            var fieldNames =  tableauConnDataObj2.fnArray; 
                            var fieldTypes =  tableauConnDataObj2.ftArray; 
                    
                        }
                        // else {          
                            // var fieldNames =  "test";
                            // var fieldTypes = "float";
                        // }
            
    tableau.headersCallback(fieldNames, fieldTypes);

    }; //end of myConnector.getColumnHeaders function
  
  
    myConnector.getTableData = function(lastRecordToken) {
    var dataToReturn = [];
    var lastRecordToken = 0;
    var hasMoreData = false;
    var tableauConnData = tableau.connectionData;
    var tableauConnDataObj = JSON.parse(tableauConnData);
    
    //ajax request for ML algorithm
    var xhr = $.ajax({
                  type: 'POST',
                  url: '/callMLalgorithm',
                  data: tableauConnData,
                  dataType: "json",     //Converts the json String into a true json object
                  contentType: 'application/json',
                  success:  function (obj) {
                  
                                //loop through tableauConnDataObj.fnArray
                                var dynObj = tableauConnDataObj.fnArray;
                                $.each(obj, function(index, objs) {
                                                var entryDyn = {};  
                                                for (var i = 0; i < dynObj.length; i++) {
                                                entryDyn[dynObj[i]] = objs[dynObj[i]];
                                                }
                                    dataToReturn.push(entryDyn);
                                                    
                                }); //end of $.each
                            
                            tableau.dataCallback(dataToReturn, lastRecordToken, false);
                    
                    }, //end success function
            
                    error: function (xhr, ajaxOptions, thrownError) {
                    tableau.log("Connection error: " + xhr.responseText + "\n" + thrownError);
                    tableau.abortWithError("Error while trying to connect to the test data source.");
                    }
        }); //end of ajax

        }; // end myConnector.getTableData
        
    tableau.registerConnector(myConnector);
        
}   //end of init

function cancelProcessData() {
    $(".uploadparameters").fadeIn(500); 
    $("#successInfo").fadeOut(200);
    $("#parameterInfo").fadeOut(200);
    $("#processFilebtn").hide();
    $("#cancelbtn").fadeOut(200);
    $(".LogisticParameter").hide();
    $("#errorInfo").hide();
}

function processData() {
    algorithmChosen = $('.algoStatus').text();
    algorithmChosen = algorithmChosen.trim();
    L1L2 = $('#L1orL2 input:radio:checked').val();
    GridSearch = $('#GridSearch input:radio:checked').val();
    numbTrees = $("#numbtrees2").val();
    splitCriteria = $('#splitCriteria input:radio:checked').val();
    submit = 0;
    
    
    // Set the text for Tableau connection
    var connectionText = "";
    if (algorithmChosen == 'Clustering(K-means)') {
        connectionText = trainingFileTxt;
    }
    else {
        connectionText = scoringFileTxt;
    }
    
    //validate the form by calling the validateUploadForm function    
    var validProcess = validateProcessForm(numbTrees);
    if (validProcess == true) { 
        tableau.connectionName = algorithmChosen + " - " + connectionText;
        tableau.connectionData = getTableauConnData(algorithmChosen);                            
        tableau.submit();
    }
    else {
        $("#successInfo").hide();
        $("#errorInfo").fadeIn(500);
        $("#errorInfo").html("<i class='fa fa-exclamation-triangle fa-2x'" + 
        'aria-hidden="true"></i> <strong>&nbsp; Error:</strong> ' + validProcess);
    }
    
}
