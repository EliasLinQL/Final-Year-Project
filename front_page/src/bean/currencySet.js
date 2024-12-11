class CurrencySet {
    constructor(name, dates, icon) {
        this.name = name;
        this.dates = dates;
        this.icon = icon;
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
    getIcon() {
        return this.icon;
    }
    setIcon(icon) {
        this.icon = icon;
    }
}

export default CurrencySet;