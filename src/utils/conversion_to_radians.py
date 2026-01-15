def conversion_to_radians(tr0, conf):

    tr = tr0.copy()

    def convertTemp(trace):
        Tvolt = trace.data * conf.get('gainTemp')
        coeff = conf.get('calcTempCoefficients')
        return coeff[0] + coeff[1]*Tvolt + coeff[2]*Tvolt**2 + coeff[3]*Tvolt**3

    def convertTilt(trace, conversion, sensitivity):
        return trace.data * conversion * sensitivity

    if tr.stats.channel[-1] == 'T':
        tr.data = convertTemp(tr)
    elif tr.stats.channel[-1] == 'N':
        tr.data = convertTilt(tr, conf['convTN'], conf['gainTilt'])
    elif tr.stats.channel[-1] == 'E':
        tr.data = convertTilt(tr, conf['convTE'], conf['gainTilt'])

    return tr.data