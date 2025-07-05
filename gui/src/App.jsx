import React from 'react'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { Toaster } from 'react-hot-toast'
import AppRoutes from './routes'
import QueryClientProvider from './components/QueryClientProvider'

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#667eea',
      light: '#8ba3f0',
      dark: '#4a5db8'
    },
    secondary: {
      main: '#764ba2',
      light: '#9b6bc7',
      dark: '#5a3678'
    },
    background: {
      default: '#0a0e1a',
      paper: '#1a1f2e'
    },
    text: {
      primary: '#ffffff',
      secondary: '#b8c5d6'
    }
  },
  typography: {
    fontFamily: 'Inter, sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem'
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem'
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.5rem'
    },
    h4: {
      fontWeight: 500,
      fontSize: '1.25rem'
    }
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'rgba(26, 31, 46, 0.8)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '16px'
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          textTransform: 'none',
          fontWeight: 500
        }
      }
    }
  }
})

function App() {
  return (
    <QueryClientProvider>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <AppRoutes />
        <Toaster 
          position="top-right"
          toastOptions={{
            style: {
              background: '#1a1f2e',
              color: '#ffffff',
              border: '1px solid rgba(255, 255, 255, 0.1)'
            }
          }}
        />
      </ThemeProvider>
    </QueryClientProvider>
  )
}

export default App