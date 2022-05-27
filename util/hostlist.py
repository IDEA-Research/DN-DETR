# A mapper from node name to ip address

def nodename_to_ip(nodename: str):
    """
    e.g. 
    Input:
        nodename: dgx020 / 020
    return: 192.168.190.20
    """
    assert len(nodename) == 3 or (len(nodename) == 6 and nodename.startswith('dgx')), \
        "invalid nodename: {}".format(nodename)
    if len(nodename) == 6:
        nodename = nodename[3:]
    return "192.168.190.{}".format(str(int(nodename)))

def parse_nodelist(nodelist: str):
    """
    e.g.
    Input:
        nodelist(SLURM_NODELIST): 
            e.g:    "dgx[001-002]", 
                    "dgx[074,076-078]"
    return 
        ['dgx001', 'dgx002'], 
        ['dgx074', 'dgx076', 'dgx077', 'dgx078']
    """
    if '[' not in nodelist:
        return [nodelist]
    nodelist = nodelist.split('[')[1].split(']')[0].strip()

    reslist = []
    blocklist = [i.strip() for i in nodelist.split(',')]
    for block in blocklist:
        if '-' in block:
            start_cnt, end_cnt = block.split('-')
            block_nodes = ["dgx%03d" % i for i in range(int(start_cnt), int(end_cnt) + 1)]
        else:
            block_nodes = ["dgx%03d" % int(block)]
        reslist.extend(block_nodes)
    return reslist




if __name__ == "__main__":
    print(parse_nodelist('dgx[074,076-078]'))