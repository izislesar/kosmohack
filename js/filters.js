function getFilters() {

    return {
        dateFrom:
            document.getElementById("date-from").value,

        dateTo:
            document.getElementById("date-to").value,

        territory:
            document.getElementById("territory").value
    };

}