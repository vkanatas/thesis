

arguments = {
    "dataset": "refit",
    "appliance": "Dishwasher",
    "dop": 0.4,
    "type": "seq2seq"
}

# Dictionary with info for the appliances Currently based on Jack Kelly Table 4
houses_ukdale = {
    "Kettle": {
        "houses": {
            "1": 10,
            "3": 2,
            "5": 18,
            "2": 8,
        },
        "dates": {
            "house1": ['2013-11-09', '2016-04-25'],
            "house2": ['2013-04-18', '2013-10-10'],
            "house3": ['2013-02-27', '2013-04-08'],
            "house5": ['2014-06-29', '2014-09-09'],

        }

    },
    "Washing Machine": {
        "houses": {
            "1": 5,
            "5": 24,
            "2": 12,
        },
        "dates": {
            "house1": ['2012-11-09', '2017-04-25'],
            "house2": ['2013-05-20', '2013-10-10'],
            "house5": ['2014-06-29', '2014-11-13'],

        }

    },
    "Dishwasher": {
        "houses": {
            "1": 6,
            "5": 22,
            "2": 13,
        },
        "dates": {
            "house1": ['2012-11-09', '2017-04-25'],
            "house2": ['2013-05-20', '2013-10-10'],
            "house5": ['2014-06-29', '2014-11-13'],

        }

    },
    "Fridge": {
        "houses": {
            "1": 12,
            "4": 5,
            "5": 19,
            "2": 14,
        },
        "dates": {
            "house1": ['2013-05-10', '2015-05-10'],  # only 2 years
            "house2": ['2013-05-20', '2013-10-10'],
            "house4": ['2013-03-09', '2013-10-01'],
            "house5": ['2014-06-29', '2014-11-13'],

        }
    },

}
houses_refit = {

    "Fridge": {
        "houses": {
            "2": 'Appliance1',
            "7": 'Appliance1',
            "12": 'Appliance1',
            "5": 'Appliance1',
            "9": 'Appliance1',
            "15": 'Appliance1',
        },

        "dates": {
            "house2": ['2014-02-18', '2014-09-30'],
            "house5": ['2013-10-02', '2014-09-29'],
            "house7": ['2014-03-11', '2014-09-29'],
            "house9": ['2014-05-03', '2014-09-29'],
            "house12": ['2014-03-07', '2014-09-30'],
            "house15": ['2013-12-18', '2014-09-22']
        }
    },
    "Dishwasher": {
        "houses": {
            "2": 'Appliance3',
            "5": 'Appliance4',
            "7": 'Appliance6',
            "9": 'Appliance4',
            "13": 'Appliance4',
            "18": 'Appliance6',
            "20": 'Appliance5'
        },

        "dates": {
            "house2": ['2014-03-22', '2014-07-22'],
            "house5": ['2013-09-27', '2014-07-22'],
            "house7": ['2013-11-05', '2014-09-22'],
            "house9": ['2013-12-18', '2014-09-21'],
            "house13": ['2014-03-05', '2014-09-06'],
            "house18": ['2014-03-07', '2014-07-20'],
            "house20": ['2014-03-20', '2014-09-30'],

        }
    },
}

appliances = {
    "Kettle": {

        "on_power": 2000,
        "max_threshold": 3100,
        "window_size": 512,
        "stride": 10,


    },
    "Washing Machine": {

        "on_power": 20,
        "max_threshold": 2500,
        "window_size": 512,
        "stride": 10,

    },
    "Fridge": {

        "on_power": 50,
        "max_threshold": 300,
        "window_size": 512,
        "stride": 10,

    },
    "Dishwasher": {

        "on_power": 10,
        "max_threshold": 2500,
        "window_size": 512,
        "stride": 10,

    },
}
