var inputFromDate = $("#fromDate"), inputToDate = $("#toDate"), buttonGetData = $("#get")

var startYear = 2000, endYear = new Date().getFullYear();

function wait(ms) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      resolve(ms)
    }, ms )
  })
}

async function downloadDataForRange(fromDate, toDate) {
	setDates(fromDate, toDate);
	$("#get").click();
	await wait(2000);
	triggerDownload();
	await wait(2000);
}

function setDates(fromDate, toDate) {
	setDate(fromDate, inputFromDate)
	setDate(toDate, inputToDate)
}

function setDate(dateVal, input) {
	input.val(`${('0'+dateVal.getDate()).slice(-2)}-${('0'+(dateVal.getMonth()+1)).slice(-2)}-${dateVal.getFullYear()}`);
}

function triggerDownload() {
    let downloadLink = $(".download-data-link a")
    if (downloadLink.length) {
	    downloadLink[0].click()
    }
}

async function downloadDataForAYear(forYear) {
	let startYearDate = new Date(forYear, 0, 1);
	let firstHalfEndDate = new Date(forYear, 6, 0);
	let secondHalfStartDate = new Date(forYear, 6, 1);
	let yearEndDate = new Date(forYear, 12, 0);
	
	await downloadDataForRange(startYearDate, firstHalfEndDate);
	await downloadDataForRange(secondHalfStartDate, yearEndDate);
}

async function downloadAnnualData(startYear) {
	for(let curYear = startYear; curYear < new Date(Date.now()).getFullYear(); curYear++) {
		await downloadDataForAYear(curYear)
	}
}