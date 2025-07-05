import React from 'react'
import { 
  Card, 
  CardContent, 
  Typography, 
  Box,
  Chip,
  LinearProgress,
  IconButton
} from '@mui/material'
import { 
  PlayArrow, 
  Pause, 
  Schedule, 
  CheckCircle,
  Error,
  Assignment
} from '@mui/icons-material'
import { motion } from 'framer-motion'

const TaskQueue = () => {
  const tasks = [
    {
      id: 'task_1',
      title: 'Market Research Analysis',
      status: 'running',
      progress: 75,
      priority: 'high',
      agent: 'Research Assistant',
      eta: '5 min'
    },
    {
      id: 'task_2',
      title: 'Email Response Generation',
      status: 'queued',
      progress: 0,
      priority: 'medium',
      agent: 'Communication Hub',
      eta: '2 min'
    },
    {
      id: 'task_3',
      title: 'Code Refactoring',
      status: 'running',
      progress: 45,
      priority: 'low',
      agent: 'Code Editor',
      eta: '15 min'
    },
    {
      id: 'task_4',
      title: 'Data Backup Process',
      status: 'completed',
      progress: 100,
      priority: 'medium',
      agent: 'System Manager',
      eta: 'Done'
    },
    {
      id: 'task_5',
      title: 'AI Model Training',
      status: 'failed',
      progress: 30,
      priority: 'high',
      agent: 'ML Trainer',
      eta: 'Failed'
    }
  ]

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return '#4ecdc4'
      case 'queued': return '#ffa726'
      case 'completed': return '#4caf50'
      case 'failed': return '#ff6b6b'
      default: return '#95a5a6'
    }
  }

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high': return '#ff6b6b'
      case 'medium': return '#ffa726'
      case 'low': return '#4ecdc4'
      default: return '#95a5a6'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running': return <PlayArrow sx={{ fontSize: 16 }} />
      case 'queued': return <Schedule sx={{ fontSize: 16 }} />
      case 'completed': return <CheckCircle sx={{ fontSize: 16 }} />
      case 'failed': return <Error sx={{ fontSize: 16 }} />
      default: return <Assignment sx={{ fontSize: 16 }} />
    }
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h5" sx={{ color: 'white', mb: 3, fontWeight: 600 }}>
          📋 Task Queue
        </Typography>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxHeight: '400px', overflow: 'auto' }}>
          {tasks.map((task, index) => (
            <motion.div
              key={task.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <Box sx={{ 
                p: 2,
                borderRadius: '12px',
                background: 'rgba(255, 255, 255, 0.05)',
                border: `1px solid ${getStatusColor(task.status)}40`,
                transition: 'all 0.3s ease',
                '&:hover': {
                  background: 'rgba(255, 255, 255, 0.08)',
                  border: `1px solid ${getStatusColor(task.status)}60`
                }
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Box sx={{ 
                    color: getStatusColor(task.status),
                    mr: 1,
                    display: 'flex',
                    alignItems: 'center'
                  }}>
                    {getStatusIcon(task.status)}
                  </Box>
                  
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="subtitle2" sx={{ color: 'white', fontWeight: 600 }}>
                      {task.title}
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#b8c5d6', fontSize: '0.8rem' }}>
                      {task.agent}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Chip
                      label={task.priority.toUpperCase()}
                      size="small"
                      sx={{
                        background: `${getPriorityColor(task.priority)}20`,
                        color: getPriorityColor(task.priority),
                        fontWeight: 600,
                        fontSize: '0.65rem'
                      }}
                    />
                    
                    <Typography variant="body2" sx={{ color: '#b8c5d6', fontSize: '0.8rem' }}>
                      {task.eta}
                    </Typography>
                  </Box>
                </Box>

                {task.status === 'running' && (
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" sx={{ color: '#b8c5d6', fontSize: '0.8rem' }}>
                        Progress
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'white', fontSize: '0.8rem' }}>
                        {task.progress}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={task.progress}
                      sx={{
                        height: 4,
                        borderRadius: 2,
                        background: 'rgba(255, 255, 255, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          background: `linear-gradient(90deg, ${getStatusColor(task.status)}80, ${getStatusColor(task.status)})`
                        }
                      }}
                    />
                  </Box>
                )}

                {task.status === 'failed' && (
                  <Box sx={{ 
                    mt: 1,
                    p: 1,
                    borderRadius: '8px',
                    background: 'rgba(255, 107, 107, 0.1)',
                    border: '1px solid rgba(255, 107, 107, 0.3)'
                  }}>
                    <Typography variant="body2" sx={{ color: '#ff6b6b', fontSize: '0.8rem' }}>
                      Task failed: Connection timeout. Retry scheduled.
                    </Typography>
                  </Box>
                )}
              </Box>
            </motion.div>
          ))}
        </Box>

        <Box sx={{ mt: 3, p: 2, borderRadius: '12px', background: 'rgba(102, 126, 234, 0.1)' }}>
          <Typography variant="body2" sx={{ color: 'white', fontWeight: 600, mb: 1 }}>
            Queue Statistics
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
              Active: 2 | Queued: 1 | Completed: 1
            </Typography>
            <Typography variant="body2" sx={{ color: '#4ecdc4' }}>
              80% Success Rate
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  )
}

export default TaskQueue