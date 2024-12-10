class CurrencySet {
    constructor(name, dates) {
        this.name = name;
        this.dates = dates;
    }

    getName() {
        return this.name;
    }
    setName(name) {
        this.name = name;
    }
    getDates() {
        return this.dates;
    }
    setDates(dates) {
        this.dates = dates;
    }
}

export default CurrencySet;