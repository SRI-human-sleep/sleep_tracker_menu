import os


def save_directory_generation(
        save_path: str,
        dir_to_create: str,
        subdirs: list = None
) -> list:
    """

        Parameters
        ----------
        save_path : str
            main directory for saving data
        dir_to_create : str

        subdirs : list
            list of devices that undergo the performance evaluation.


        Returns
        -------
        list
            saving directory and its subdirectories
    """
    dir_to_create = os.path.join(save_path, dir_to_create)
    if os.path.exists(dir_to_create):
        print(f'Already existing directory: {dir_to_create}')
    else:
        print(f'Creating new directory: {dir_to_create}')
        os.makedirs(dir_to_create)

    if subdirs:
        subdirs_append = []
        for i in subdirs:
            dir_cycle = os.path.join(dir_to_create, i)
            subdirs_append.append(dir_cycle)
            if os.path.exists(dir_cycle):
                print(f'Already existing directory: {dir_cycle}')
            else:
                print(f'Creating new directory: {dir_cycle}')
                os.makedirs(dir_cycle)
    else:
        subdirs_append = None

    return [dir_to_create, subdirs_append]
