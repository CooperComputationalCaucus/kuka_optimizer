class Events:
    OPTMIZATION_START = 'optmization:start'
    OPTMIZATION_STEP = 'optmization:step'
    OPTMIZATION_END = 'optmization:end'
    BATCH_END = 'batch:end'


DEFAULT_EVENTS = [
    Events.OPTMIZATION_START,
    Events.OPTMIZATION_STEP,
    Events.OPTMIZATION_END,
    Events.BATCH_END
]
