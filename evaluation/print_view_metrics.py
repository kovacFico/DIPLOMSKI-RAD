def print_single_view_metrics(id_report, coverage_report):
    """
    Ispisuje kompletan evaluacijski izvještaj u konzolu.
    
    Args:
        id_report (dict): Evaluacijski izvještaj identiteta.
        coverage_report (dict): Evaluacijski izvještaj pokrivenosti.
    """


    print("\n")
    print("=" * 60)
    print("SINGLE-VIEW IDENTITY EVALUATION")
    print("=" * 60)

    overall = id_report["overall"]

    print(
        f"Identity accuracy : "
        f"{overall['identity_accuracy'] * 100:.2f}%"
    )

    print(
        f"ID switches       : "
        f"{overall['id_switches']}"
    )

    print(
        f"Consistency       : "
        f"{overall['identity_consistency_frames']:.1f} frames "
        f"({overall['identity_consistency_seconds']:.2f} s)"
    )

    print("\nPER TRACK")
    print("-" * 60)

    for track_id, data in id_report["tracks"].items():
        print(
            f"\nTrack {track_id} "
            f"({data['identity']})"
        )

        print(
            f"  Accuracy         : "
            f"{data['identity_accuracy'] * 100:.2f}%"
        )

        print(
            f"  ID switches      : "
            f"{data['id_switches']}"
        )

        print(
            f"  Correct frames   : "
            f"{data['correct_frames']}"
        )

        print(
            f"  Wrong frames     : "
            f"{data['wrong_frames']}"
        )

        print(
            f"  Consistency      : "
            f"{data['identity_consistency_seconds']:.2f} s"
        )


    print("\nCOVERAGE")
    print("-" * 60)

    for track_id, data in coverage_report.items():
        print(
            f"Track {track_id}: "
            f"{data['coverage'] * 100:.2f}% "
            f"({data['observed_frames']}/"
            f"{data['total_frames']} frames)"
        )
    print("=" * 60)
    
    
def print_multiview_metrics(report):
    '''
    Ispisuje rezultate multi-view evaluacije u terminal.

    Prikazuje:
        - ukupni Identity Accuracy
        - ukupni broj ID Switch događaja
        - prosječnu Identity Consistency
        - Coverage
        - frame od kojeg je multi-view evaluator počeo raditi
        - korištenje pojedinog viewa
        - rezultate svakog globalnog identiteta zasebno

    Args:
        report (dict): Izvještaj generiran metodom
            MultiViewIdentityEvaluator.report().
    '''

    overall = report["overall"]
    global_tracks = report["global_tracks"]

    print("\n")
    print("=" * 65)
    print("MULTI-VIEW IDENTITY EVALUATION")
    print("=" * 65)


    print("\nOVERALL")
    print("-" * 65)

    print(
        f"Identity Accuracy : "
        f"{overall['identity_accuracy'] * 100:.2f}%"
    )

    print(
        f"ID Switches       : "
        f"{overall['id_switches']}"
    )

    print(
        f"Consistency       : "
        f"{overall['identity_consistency_frames']:.1f} frames "
        f"({overall['identity_consistency_seconds']:.2f} s)"
    )

    print(
        f"Coverage          : "
        f"{overall['coverage'] * 100:.2f}%"
    )

    print(
        f"Evaluation start  : "
        f"frame {overall['start_frame']}"
    )


    print("\nVIEW USAGE")
    print("-" * 65)

    view_usage = overall["view_usage"]

    total_view_usage = sum(
        view_usage.values()
    )

    for view_id, used_frames in view_usage.items():

        if total_view_usage == 0:
            usage_percent = 0.0

        else:
            usage_percent = (
                used_frames /
                total_view_usage
            ) * 100.0

        print(
            f"View {view_id:<2}          : "
            f"{used_frames} assignments "
            f"({usage_percent:.2f}%)"
        )


    print("\nGLOBAL IDENTITIES")
    print("-" * 65)

    for global_id, data in global_tracks.items():

        print(
            f"\nGLOBAL {global_id} "
            f"({data['identity']})"
        )

        print(
            f"  Identity Accuracy : "
            f"{data['identity_accuracy'] * 100:.2f}%"
        )

        print(
            f"  ID Switches       : "
            f"{data['id_switches']}"
        )

        print(
            f"  Correct Frames    : "
            f"{data['correct_frames']}"
        )

        print(
            f"  Wrong Frames      : "
            f"{data['wrong_frames']}"
        )

        print(
            f"  Observed Frames   : "
            f"{data['observed_frames']}"
        )

        print(
            f"  Consistency       : "
            f"{data['identity_consistency_frames']:.1f} frames "
            f"({data['identity_consistency_seconds']:.2f} s)"
        )

        print(
            f"  Coverage          : "
            f"{data['coverage'] * 100:.2f}%"
        )

    print("\n" + "=" * 65)