#!/usr/bin/env python3

import os
from os.path import join, split, isdir
import glob
import time
from collections import Counter
from pprint import pprint

# TODO enable these w/ max size either >= all replays or >= all within some
# timeframe, like a season
# TODO TODO but check how much disk space these choices will cost me!
#os.environ['SC2READER_CACHE_DIR'] = "path/to/local/cache"
#os.environ['SC2READER_CACHE_MAX_SIZE'] = 100

import sc2reader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


REPLAY_EXTENSION = 'SC2Replay'

_replay_dir = None
def replay_dir():
    global _replay_dir
    if _replay_dir is not None:
        return _replay_dir

    replay_dir = os.path.expanduser('~/Documents/StarCraft II/Accounts')

    # TODO are the two folder names i'm searching for meaningful?
    # should i save them as other variables?

    # (no attempt to support multiple accounts now)
    account_dirs = glob.glob(join(replay_dir, '*/'))
    assert len(account_dirs) == 1
    account_dir = account_dirs[0]

    # not sure how multiple subdirs here differ from above...
    # might be a "profile" / server thing
    subdirs = [d for d in glob.glob(join(account_dir, '*/'))
        if not os.path.dirname(d).endswith('Hotkeys')
    ]
    assert len(subdirs) == 1
    # This seems equal to my player.toon_handle parseable at <= load_level=2
    # w/ sc2reader (though still not sure what that means)
    subdir = subdirs[0]

    rdir = join(subdir, 'Replays', 'Multiplayer')
    assert isdir(rdir)
    _replay_dir = rdir
    return rdir


def list_replays(under=None, recursive=True):
    """Lists all replays, or all in folder `under`, if passed.
    """
    if under is None:
        under = replay_dir()
    
    replay_glob = f'*.{REPLAY_EXTENSION}'
    if recursive:
        replay_glob = f'**/{replay_glob}'

    # TODO does passing recursive=True here cause the '**/...' glob expression
    # to behave the same in python <3.5 and python >=3.5? (was intention)
    replays = glob.glob(join(under, replay_glob), recursive=recursive)
    return replays


# At a minimum, this is probably not goign to be correct going back to games
# where there were different leagues available.
highest_league_num2str = {
    1: 'bronze',
    2: 'silver',
    3: 'gold',
    4: 'platinum',
    5: 'diamond',
    6: 'master',
    7: 'grandmaster',
    8: 'no_ranking_this_season',
    # TODO what is 0? check some bnet profiles to see?
    0: 'not_sure'
}


free_units = {
    'broodling',
    'locust',
    'interceptor'
}
# TODO deal w/ units modes, as seems necessary
# (warpprismphasing and maybe something similar for siege?)
_real_army_units_seen = set()
def real_army_units(units):
    rau = [u for u in units if u.is_army and not u.hallucinated
        and u.title != 'Overlord' and u.title.lower() not in free_units
    ]
    for u in rau:
        _real_army_units_seen.add(u)
    return rau


unit2abbreviation = {
    # TODO test these keys are all correct and not missing spaces or something
    'darktemplar': 'DT',
    'zergling': 'ling',
    'mutalisk': 'muta',
    'immortal': 'immo',
    'warpprism': 'prism'
}
def abbreviate_unit(unit):
    unit = unit.lower()
    if unit in unit2abbreviation:
        return unit2abbreviation[unit]
    else:
        return unit


def count_army_units(army_units):
    # Throwing away the time information, etc.
    army_units = [abbreviate_unit(u.title) for u in army_units]
    return Counter(army_units)


def most_common_army_unit(army_units):
    if len(army_units) == 0:
        return None
    counter = count_army_units(army_units)
    return counter.most_common(1)[0][0]


# TODO maybe include upgrades in something like this / in another str
def army_summary_str(army_units, scale_by='num_units', n=3):
    if len(army_units) == 0:
        return None
    counter = count_army_units(army_units)

    unit_names = []
    counts = []
    unit_name2idx = dict()
    for i, (u, c) in enumerate(counter.items()):
        unit_names.append(u)
        counts.append(c)
        unit_name2idx[u] = i
    counts = np.array(counts)

    if scale_by == 'num_units':
        percentages = counts / np.sum(counts)
    # TODO TODO TODO weight by supply!
    # TODO maybe also consider weighting by or resources?
    else:
        raise NotImplementedError(f'scale_by={scale_by} not supported')

    # TODO maybe some elbow finding method to pick n highest to use?

    summary_str_parts = []
    remaining = 1.0
    for u, _ in counter.most_common(n):
        i = unit_name2idx[u]
        p = percentages[i]
        remaining -= p
        summary_str_parts.append(f'{p:.0%} {u}')

    delimiter = ', '
    summary_str = delimiter.join(summary_str_parts)
    if remaining > 0.03:
        summary_str += f' ({remaining:.0%} other)'

    return summary_str


# TODO TODO how to restrict all replays to just 1v1 ranked (or just 1v1 ranked /
# unranked, if that's much easier)?
def load_1v1_ranked(paths, my_name, analyze_units=True):
    # TODO maybe just use load_players or something else to just get the data i
    # want, rather than load_level=2, which also loads other stuff
    # (though idk if any of other stuff totals to significant time loading...)

    # Technically possible to load all at load_level=1 first and check
    # .type == '1v1', but 1) most of my games are 1v1 anyway and
    # 2) loading all at load_level=1 or load_level=2 seems to take about the
    # same amount of time, so unlikely that any such strategy would help much,
    # as far as total load time is concerned.

    if analyze_units:
        load_level = 3
    else:
        load_level = 2

    before = time.time()

    print(f'Loading {len(paths)} replays at load_level={load_level} ...')
    replays = sc2reader.load_replays(paths, load_level=load_level)

    shortcircuit = True
    n_with_computers = 0
    n_ladder = 0
    n_1v1 = 0
    n_competitive = 0 

    # If the value for a key is `None` here, the variable is assumed to be
    # accessible as an attribute of the replay object.
    game_vars2getters = {
        'filename': None,
        'start_time': None,
        # TODO matter whether i use this .real_length over just .length?
        'duration_s': lambda r: r.real_length.total_seconds(),
        'map_name': None,
        # The list of players indexed here is the winning team, which should
        # always just have one person given the restrictions earlier in this
        # function.
        # TODO elsewhere assert name is unique in 1v1
        'won': lambda r: r.winner.players[0] == my_name,
        'expansion': None,
        # TODO could also get maybe build / base_build, though not sure if they
        # map to the balance patches i might care about or how to do that
        # mapping
    }
    # TODO TODO get variable representing the season
    my_prefix = 'my_'
    opponent_prefix = 'opponent_'
    # If the value for a key is `None` here, the variable is assumed to be
    # accessible as an attribute of the player (`sc2reader.objects.Participant`)
    # object.
    my_vars2getters = {
        'pick_race': None,
        'play_race': None,
        # this is just to see if variation in this for my opponents is maybe
        # just b/c the mapping changed and thus my subregion also changed
        'subregion': None,
        # MMR before match. Not sure if "scaled" actually has any meaning here.
        'scaled_rating': lambda p: p.init_data['scaled_rating'],
    }
    opponent_vars2getters = {
        'pick_race': None,
        'play_race': None,
        'name': None,
        'clan_tag': None,
        # TODO get which sub league they are in? that vary, or do i always play
        # people within mine?
        # TODO does subregion actually vary for my opponents? drop if not.
        'subregion': None,
        # TODO are other IDs (like battle tag) avaible directly from replay?
        # Can be used to generate URL to their BNet profile
        # (directly accessible w/ player.url)
        'bnet_uid': lambda o: o.detail_data['bnet']['uid'],
        'highest_league': lambda p: highest_league_num2str[
            getattr(p, 'highest_league')
        ],
        'scaled_rating': lambda p: p.init_data['scaled_rating'],
        # Chat will be handled separately.
    }

    # Has one key for each in the three dicts above (though player variables
    # will be prefixed), and the values for each should be same-length lists
    # of length equal to the number of replays analyzed.
    var2value_lists = {v: [] for v in (
        list(game_vars2getters.keys()) +
        [my_prefix + k for k in my_vars2getters.keys()] +
        [opponent_prefix + k for k in opponent_vars2getters.keys()] +
        [p + 'chat' for p in (my_prefix, opponent_prefix)]
    )}
    if analyze_units:
        var2value_lists[my_prefix + 'mode_army_unit'] = []
        var2value_lists[opponent_prefix + 'mode_army_unit'] = []
        var2value_lists[my_prefix + 'army_summary'] = []
        var2value_lists[opponent_prefix + 'army_summary'] = []

    # TODO need to separately filter out custom 1v1 games against humans,
    # or do is_ladder / competitive already effectively filter that?
    # (look at whats get filtered and check for my recent games w/ brian for
    # instance...)
    for replay in tqdm(replays, total=len(paths)):
        if len(replay.computers) > 0:
            n_with_computers += 1
            if shortcircuit:
                continue

        # TODO seems ladder might already only select 1v1s?
        if not replay.is_ladder:
            if shortcircuit:
                continue
        else:
            assert replay.competitive
            n_ladder += 1

        # TODO players include AI? (if not, this might catch non-1v1 games...)
        if len(replay.players) != 2:
            if shortcircuit:
                continue
        else:
            assert replay.type == '1v1'
            n_1v1 += 1

        # TODO i have definitely played *some* 1v1 unranked games, so it seems
        # this must be True in those cases as well...
        assert replay.competitive
        n_competitive += 1

        # TODO why is this always None? what is this for?
        # TODO check if when fully loaded
        assert replay.ranked is None

        assert replay.players[0].name != replay.players[1].name
        if replay.players[0].name == my_name:
            my_idx = 0
            opponent_idx = 1
        elif replay.players[1].name == my_name:
            my_idx = 1
            opponent_idx = 0
        else:
            raise ValueError('no player with name matching my_name={my_name}')
        me = replay.players[my_idx]
        opponent = replay.players[opponent_idx]

        for var, getter in game_vars2getters.items():
            if getter is None:
                value = getattr(replay, var)
            else:
                value = getter(replay)
            var2value_lists[var].append(value)

        for var, getter in my_vars2getters.items():
            if getter is None:
                value = getattr(me, var)
            else:
                value = getter(me)
            var2value_lists[my_prefix + var].append(value)

        for var, getter in opponent_vars2getters.items():
            if getter is None:
                value = getattr(opponent, var)
            else:
                value = getter(opponent)
            var2value_lists[opponent_prefix + var].append(value)

        # Handling chat separately cause it's not most easily accessible as a fn
        # of the player objects it seems.
        my_chat = []
        their_chat = []
        for m in replay.messages:
            if not m.to_all:
                continue

            # might want to also store m.frame to re-order later
            # (if i want that)
            if m.player.name == my_name:
                my_chat.append(m.text)
            else:
                assert m.player.name == opponent.name
                their_chat.append(m.text)

        var2value_lists[my_prefix + 'chat'].append(my_chat)
        var2value_lists[opponent_prefix + 'chat'].append(their_chat)

        if analyze_units:
            opponent_army_units = real_army_units(opponent.units)
            opponent_mode_army_unit = most_common_army_unit(opponent_army_units)
            opponent_army_summary = army_summary_str(opponent_army_units)
            var2value_lists[opponent_prefix + 'mode_army_unit'].append(
                opponent_mode_army_unit
            )
            var2value_lists[opponent_prefix + 'army_summary'].append(
                opponent_army_summary
            )

            my_army_units = real_army_units(me.units)
            my_mode_army_unit = most_common_army_unit(my_army_units)
            my_army_summary = army_summary_str(my_army_units)
            var2value_lists[my_prefix + 'mode_army_unit'].append(
                my_mode_army_unit
            )
            var2value_lists[my_prefix + 'army_summary'].append(my_army_summary)

    '''
    print('n_1v1:', n_1v1)
    print('n_with_computers:', n_with_computers)
    print('n_ladder:', n_ladder)
    print('n_competitive:', n_competitive)
    '''
    # TODO why does even load_level=4 seem to not correctly specify
    # replay.ranked? need to do some other init? is that the flag i want?
    # TODO how does .ranked differ from .competitive?

    # TODO compare what i can get out of 3 vs 4.
    '''
    for path in tqdm(paths_to_fully_load, total=len(paths_to_fully_load)):
        r3 = sc2reader.load_replay(path, load_level=3)
        r4 = sc2reader.load_replay(path, load_level=4)
        # player.units something i want here?
        import ipdb; ipdb.set_trace()
    '''

    # TODO TODO would it take too much memory to store full load_level 3/4
    # replay objects to each? store some more compact repr of build orders
    # in a column too?
    '''
    load_level = 4
    print(f'Loading {len(paths)} replays at load_level={load_level} ...')
    l4_replays = sc2reader.load_replays(paths, load_level=load_level)
    '''

    # TODO TODO maybe derive columns for whether i used hotkeys for various
    # things (particularly (all/most of) my bases)

    total_s = time.time() - before
    print(f'Loading took {total_s:.0f}s')

    df = pd.DataFrame(var2value_lists)

    n_replays_before = len(df)
    df.drop_duplicates(subset=['start_time'], inplace=True)
    n_dropped = n_replays_before - len(df)
    if n_dropped > 0:
        print(f'Dropped {n_dropped} replays with duplicate start_time')

    df.set_index('start_time', inplace=True)
    df.sort_index(inplace=True)
    return df


def game_fraction_familiar_opponent(df):
    """
    Assumes `df` rows are already sorted by `start_time`, with older games
    in earlier rows.
    """
    played_opponent_before = df['opponent_bnet_uid'].duplicated(keep='first')
    frac_familiar_series = \
        played_opponent_before.cumsum() / (np.arange(len(df)) + 1)

    # TODO maybe inspect final element too

    # TODO does matplotlib work in WSL? seems like it might not...
    # TODO TODO TODO fix plotting
    #plt.plot(frac_familiar_series)
    #plt.show()

    # TODO varying lookbehind windows? (i initially described a lookbehind
    # window that extends back to time 0=first replay, but maybe try some that
    # are like weeks / months long?)

    return frac_familiar_series


# TODO TODO TODO try to group / cluster builds into a set of existing ones (or
# learn the clusters from the data) (but maybe try clustering into some popular
# spawningtool builds for instance?) (maybe just start w/ a variable that is the
# most common non-worker unit they built?)

# TODO TODO TODO maybe come up w/ some statistic representing how aggressive a
# certain player is in a certain game? (maybe compute something like how many
# minutes until first army unit sent across / attacking structure built?) or
# some statistic that tries to compute time to first peak in opponent resources
# lost (though defending would also increase this...)?  when they build first
# expansion (if they do) / early tech structures that might indicate aggression?

# TODO TODO see whether any of the statistics in the spawningtool code seem
# useful for my analysis


def main():
    paths = list_replays()

    '''
    path = paths[0]
    # load_level=4 is the default (loads everything except map, which has a
    # separate flag to load)
    # load_level=2 is the minimum that has players
    r0 = sc2reader.load_replay(path, load_level=0)
    r1 = sc2reader.load_replay(path, load_level=1)
    r2 = sc2reader.load_replay(path, load_level=2)
    # r3 also loads "tracker" events, which might not be that useful alone?
    # (or could be faster way to get a lot of the game state i care about...)
    r4 = sc2reader.load_replay(path, load_level=4)
    '''
    #import ipdb; ipdb.set_trace()

    my_name = 'TheTossBoss'
    # TODO TODO TODO cache this df (particularly if it ends up using
    # load_level=4 to get build order info)
    # TODO though see what caching is available in sc2reader, as it seems they
    # have some (see commented env var stuff above sc2reader import)
    df = load_1v1_ranked(paths, my_name)

    frac_familiar = game_fraction_familiar_opponent(df)

    print('"real" army units seen:')
    pprint(_real_army_units_seen)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

