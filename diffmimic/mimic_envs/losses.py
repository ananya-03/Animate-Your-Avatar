from diffmimic.utils.rotation6d import quaternion_to_rotation_6d


def loss_l1_relpos(qp, ref_qp):
    pos, ref_pos = qp.pos[:-1], ref_qp.pos[:-1]
    relpos, ref_relpos = (pos - pos[0])[1:], (ref_pos - ref_pos[0])[1:]
    relpos_loss = (((relpos - ref_relpos) ** 2).sum(-1) ** 0.5).mean()
    relpos_loss = relpos_loss
    return relpos_loss


def loss_l2_pos(qp, ref_qp, reduce='mean'):
    pos, ref_pos = qp.pos[:-1], ref_qp.pos[:-1]
    pos_loss = ((pos - ref_pos) ** 2).sum(-1)
    if reduce == 'mean':
        pos_loss = pos_loss.mean()
    else:
        pos_loss = pos_loss.sum()
    return pos_loss


def loss_l1_pos(qp, ref_qp):
    pos, ref_pos = qp.pos[:-1], ref_qp.pos[:-1]
    pos_loss = (((pos - ref_pos) ** 2).sum(-1)**0.5).mean()
    return pos_loss


def loss_l2_rot(qp, ref_qp, reduce='mean'):
    rot, ref_rot = quaternion_to_rotation_6d(qp.rot[:-1]), quaternion_to_rotation_6d(ref_qp.rot[:-1])
    rot_loss = ((rot - ref_rot) ** 2).sum(-1)
    if reduce == 'mean':
        rot_loss = rot_loss.mean()
    else:
        rot_loss = rot_loss.sum()
    return rot_loss


def loss_l2_vel(qp, ref_qp, reduce='mean'):
    vel, ref_vel = qp.vel[:-1], ref_qp.vel[:-1]
    vel_loss = ((vel - ref_vel) ** 2).sum(-1)
    if reduce == 'mean':
        vel_loss = vel_loss.mean()
    else:
        vel_loss = vel_loss.sum()
    return vel_loss


def loss_l2_ang(qp, ref_qp, reduce='mean'):
    ang, ref_ang = qp.ang[:-1], ref_qp.ang[:-1]
    ang_loss = ((ang - ref_ang) ** 2).sum(-1)
    if reduce == 'mean':
        ang_loss = ang_loss.mean()
    else:
        ang_loss = ang_loss.sum()
    return ang_loss


def loss_l1_rot(qp, ref_qp):
    rot, ref_rot = quaternion_to_rotation_6d(qp.rot[:-1]), quaternion_to_rotation_6d(ref_qp.rot[:-1])
    rot_loss = (((rot - ref_rot) ** 2).sum(-1)**0.5).mean()
    return rot_loss
