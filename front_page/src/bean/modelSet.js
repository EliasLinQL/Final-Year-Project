class ModelSet {
    constructor(name, currencies) {
        this.name = name;
        this.currencies = currencies;
    }

    getName() {
        return this.name;
    }
    setName(name) {
        this.name = name;
    }
    getCurrencies() {
        return this.currencies;
    }
    setCurrencies(currencies) {
        this.currencies = currencies;
    }
}

export default ModelSet;